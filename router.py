"""Tool Router powered by Laya System 1 Decision Engine.

Implements hierarchical two-level routing and Top-10 selection:
1. Daily tool categorization (24h TTL).
2. Level 1: Context gatekeeping (execute_tools vs direct_chat) and Category selection.
   - If confident direct_chat (probability >= threshold and confident): strips tools (tools = []).
   - If uncertain or execute_tools: selects top tool categories and candidates.
3. Level 2: Top-N tool ranking for the chosen category and always_include preservation.
4. Shadow mode: performs full Laya query, response processing, and logging,
   but leaves the primary model toolset completely unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .categorizer import ToolCategorizer, extract_tool_desc, extract_tool_name
    from .client import LayaClient
    from .config import ENRICH_TOP_N, LayaConfig
    from .hygiene import looks_trivial, redact_sensitive, unwrap
    from .logger import LayaAuditLogger
    from .tool_collector import (
        bridge_tool_names,
        collect_all_tools,
        get_bridge_tools,
        is_bridge_tool,
        messages_used_bridge_tools,
        verify_bridge_tools,
    )
except ImportError:
    from categorizer import ToolCategorizer, extract_tool_desc, extract_tool_name
    from client import LayaClient
    from config import ENRICH_TOP_N, LayaConfig
    from hygiene import looks_trivial, redact_sensitive, unwrap
    from logger import LayaAuditLogger
    from tool_collector import (
        bridge_tool_names,
        collect_all_tools,
        get_bridge_tools,
        is_bridge_tool,
        messages_used_bridge_tools,
        verify_bridge_tools,
    )

logger = logging.getLogger("hermes.plugin.laya.router")

NONE_CATEGORY_ID = "none"
MAX_TRACKED_SESSIONS = 1000


def merge_tools_unique(*groups: Optional[List[Any]]) -> List[Any]:
    """Concatenate tool lists keeping the first definition of every tool name."""
    seen: Set[str] = set()
    merged: List[Any] = []
    for group in groups:
        for tool in group or []:
            name = extract_tool_name(tool)
            if name not in seen:
                seen.add(name)
                merged.append(tool)
    return merged


class ToolRouter:
    """Two-level hierarchical tool router with shadow mode support."""

    def __init__(
        self,
        laya_client: LayaClient,
        config: LayaConfig,
        categorizer: Optional[ToolCategorizer] = None,
        audit_logger: Optional[LayaAuditLogger] = None,
    ) -> None:
        self.laya = laya_client
        self.config = config
        self.categorizer = categorizer or ToolCategorizer(
            laya_client=laya_client,
            cache_path=config.cache_path,
            interval_ms=config.categorization_interval_ms,
        )
        self.audit_logger = audit_logger or LayaAuditLogger(
            log_path=config.log_path,
            min_level=config.log_level,
        )

        self.last_decision: Optional[Dict[str, Any]] = None
        self._decision_cache: "OrderedDict[str, Tuple[List[Any], Dict[str, Any]]]" = OrderedDict()

        # Progressive-disclosure (tool_search/tool_describe/tool_call) bookkeeping.
        self.bridge_status: Optional[Dict[str, Any]] = None
        self._bridge_by_session: Dict[str, List[Any]] = {}
        self._bridge_reported: Set[str] = set()
        self._unlocked_sessions: Set[str] = set()

    # ------------------------------------------------------------------ bridge tools

    def unlock_session(self, session_id: str) -> None:
        """Remember that the model used discovery tools: it gets the full tool list from now on."""
        if len(self._unlocked_sessions) > MAX_TRACKED_SESSIONS:
            self._unlocked_sessions.clear()
        self._unlocked_sessions.add(session_id or "")

    def is_unlocked(self, session_id: str) -> bool:
        return (session_id or "") in self._unlocked_sessions

    def forget_session(self, session_id: str) -> None:
        """Drop per-session bridge state (called when the Hermes session ends)."""
        key = session_id or ""
        self._unlocked_sessions.discard(key)
        self._bridge_by_session.pop(key, None)
        self._bridge_reported.discard(key)

    def check_bridge_tools(
        self,
        tools: List[Any],
        original_tools: Optional[List[Any]],
        session_id: str,
        turn_id: str,
    ) -> Dict[str, Any]:
        """Resolve the standard Hermes bridge tools and report (once per session) if they are gone.

        Sources, in order: the current tool list, the session's original list, the schemas
        remembered for this session, and the schemas Hermes itself builds.
        """
        key = session_id or ""
        remembered = self._bridge_by_session.get(key)
        status = verify_bridge_tools(
            tools,
            fallback_sources=[original_tools, remembered],
            deferred_count=max(len(original_tools or []) - len(tools), 1),
        )
        if status["bridge_tools"]:
            if len(self._bridge_by_session) > MAX_TRACKED_SESSIONS:
                self._bridge_by_session.clear()
            self._bridge_by_session[key] = status["bridge_tools"]
        self.bridge_status = status

        problems: List[str] = []
        if status["hermes_available"]:
            if status["unresolved"]:
                problems.append(f"missing standard Hermes bridge tools: {status['unresolved']}")
            problems.extend(status["api_issues"])
        if problems and key not in self._bridge_reported:
            if len(self._bridge_reported) > MAX_TRACKED_SESSIONS:
                self._bridge_reported.clear()
            self._bridge_reported.add(key)
            message = "; ".join(problems)
            logger.error(
                "[Laya] Hermes progressive-tool-disclosure check failed: %s. "
                "Hermes may have changed - the plugin %s.",
                message,
                "falls back to the full tool list" if self._should_fail_open(status) else "keeps routing as is",
            )
            self.audit_logger.log_interaction(
                session_id=session_id,
                turn_id=turn_id,
                mode=self.config.mode,
                decision="bridge_tools_missing",
                all_available_tools=[extract_tool_name(t) for t in tools],
                raw_answers={"kind": "bridge_check", "problems": problems},
            )
        return status

    @staticmethod
    def _should_fail_open(status: Dict[str, Any]) -> bool:
        """Inside Hermes, bridge tools that cannot be obtained mean the tool-hiding mechanism changed."""
        return bool(status.get("unresolved")) and bool(status.get("hermes_available"))

    def build_full_access_tools(
        self,
        tools: List[Any],
        agent: Any = None,
        bridge_tools: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Full uncollapsed catalog (core + deferred/MCP/plugin) plus the bridge tools."""
        full = collect_all_tools(agent=agent, extra_tools=tools, include_bridge_tools=True)
        return merge_tools_unique(full, tools, bridge_tools)

    def _full_access_result(
        self,
        tools: List[Any],
        agent: Any,
        bridge_tools: List[Any],
        reason: str,
        session_id: str,
        turn_id: str,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        effective = self.build_full_access_tools(tools, agent, bridge_tools)
        names = [extract_tool_name(t) for t in effective]
        decision = {
            "decision": "bridge_unlocked",
            "category": None,
            "probability": 1.0,
            "selected_tools": names,
            "exposed_tools": names,
            "shadow": False,
            "fallback_reason": reason,
            "error": None,
        }
        self.last_decision = decision
        self.audit_logger.log_interaction(
            session_id=session_id,
            turn_id=turn_id,
            mode=self.config.mode,
            decision="bridge_unlocked",
            selected_tools=names,
            all_available_tools=names,
            state_snippet=f"full tool list exposed: {reason}",
        )
        return effective, decision

    def route_tools(
        self,
        tools: List[Any],
        context: Any,
        session_id: str = "",
        turn_id: str = "",
        agent: Any = None,
        original_tools: Optional[List[Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Route tools for the upcoming model call, always keeping Hermes' discovery tools.

        Wraps :meth:`_route_tools_core` with the progressive-disclosure guarantees:
        * ``tool_search``/``tool_describe``/``tool_call`` are always part of the result
          (active modes), whatever Laya picked;
        * their presence is verified; if Hermes no longer provides them the plugin reports an
          error and falls back to the full tool list instead of hiding tools;
        * once the model has used a discovery tool in this session, it gets the full tool list.

        Args:
            original_tools: The session's standard tool list as Hermes built it (before this plugin
                rewrote ``agent.tools``); source for the bridge tool schemas.
        """
        if not self.config.enabled or self.config.is_killed or not tools:
            return self._route_tools_core(tools, context, session_id, turn_id, agent)

        status = self.check_bridge_tools(tools, original_tools, session_id, turn_id)
        bridge_tools = status["bridge_tools"]

        if not self.config.is_shadow:
            reason = None
            if self._should_fail_open(status):
                reason = "bridge_tools_missing"
            elif self.is_unlocked(session_id):
                reason = "bridge_tool_called"
            elif isinstance(context, dict) and messages_used_bridge_tools(
                context.get("conversation_history") or context.get("messages")
            ):
                self.unlock_session(session_id)
                reason = "bridge_tool_called"
            if reason:
                return self._full_access_result(tools, agent, bridge_tools, reason, session_id, turn_id)

        effective, decision = self._route_tools_core(tools, context, session_id, turn_id, agent)
        if self.config.is_shadow:
            return effective, decision

        effective = merge_tools_unique(effective, bridge_tools)
        decision["exposed_tools"] = [extract_tool_name(t) for t in effective]
        decision["bridge_tools"] = [extract_tool_name(t) for t in bridge_tools]
        return effective, decision

    def _route_tools_core(
        self,
        tools: List[Any],
        context: Any,
        session_id: str = "",
        turn_id: str = "",
        agent: Any = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Route and filter tools for the upcoming model call across the full tools catalog.

        Args:
            tools: Available tool schemas from Hermes agent.
            context: Prompt string, messages list, or TurnContext.
            session_id: Current Hermes session identifier.
            turn_id: Current Hermes turn identifier.
            agent: Optional live agent instance with full toolset scopes.

        Returns:
            Tuple of (effective_tools, decision_info).
            In shadow mode, effective_tools is always the original tools list.
        """
        if not self.config.enabled or self.config.is_killed or not tools:
            decision = {
                "decision": "killed" if self.config.is_killed else ("disabled" if not self.config.enabled else "no_tools"),
                "category": None,
                "probability": 1.0,
                "selected_tools": [extract_tool_name(t) for t in tools],
                "shadow": self.config.is_shadow,
            }
            self.last_decision = decision
            return tools, decision

        state_text = self.extract_context_state(context)
        if not state_text:
            decision = {
                "decision": "empty_context",
                "category": None,
                "probability": 1.0,
                "selected_tools": [extract_tool_name(t) for t in tools],
                "shadow": self.config.is_shadow,
            }
            self.last_decision = decision
            return tools, decision

        # 1. Fast Ack / Trivial Bypass (<1ms, zero network calls to Laya)
        if self.config.trivial_bypass and looks_trivial(state_text):
            logger.debug("Fast trivial bypass triggered for: %s", state_text[:50])
            if self.config.is_enrich_top3:
                # Enrichment never strips: the standard tool list stays exactly as it is.
                decision = {
                    "decision": "enrich_top3",
                    "category": None,
                    "probability": 1.0,
                    "selected_tools": [],
                    "added_tools": [],
                    "shadow": False,
                    "trivial_bypass": True,
                    "latency_ms": 0.1,
                }
                self.last_decision = decision
                return tools, decision

            decision = {
                "decision": "direct_chat",
                "category": None,
                "probability": 1.0,
                "selected_tools": [],
                "shadow": self.config.is_shadow,
                "trivial_bypass": True,
                "latency_ms": 0.1,
            }
            if self.config.is_shadow:
                decision["simulated_selected_tools"] = []
                self.last_decision = decision
                return tools, decision

            self.last_decision = decision
            return [], decision

        # Collect complete uncollapsed tools catalog (hot + deferred/catalog/MCP)
        full_catalog = collect_all_tools(agent=agent, extra_tools=tools)
        routing_tools = full_catalog if full_catalog else tools
        bridge_tools = get_bridge_tools(tools)

        # 2. Repeat-Ask LRU Cache
        cache_key = None
        if self.config.cache_repeat_asks:
            tools_sig = ",".join(sorted(extract_tool_name(t) for t in routing_tools))
            if self.config.is_enrich_top3:
                # The enriched result depends on the standard list it is added to.
                tools_sig += "|" + ",".join(sorted(extract_tool_name(t) for t in tools))
            cache_key = hashlib.sha256(f"{state_text}:{tools_sig}:{self.config.mode}".encode("utf-8")).hexdigest()
            if cache_key in self._decision_cache:
                cached_tools, cached_dec = self._decision_cache[cache_key]
                self._decision_cache.move_to_end(cache_key)
                dec_copy = dict(cached_dec)
                dec_copy["cached"] = True
                dec_copy["latency_ms"] = 0.0
                self.last_decision = dec_copy
                logger.debug("Reusing cached routing decision for key %s", cache_key[:8])
                effective = tools if self.config.is_shadow else cached_tools
                return effective, dec_copy

        start_time = time.time()
        try:
            if self.config.is_enrich_top3:
                selected_tools, decision = self._route_enrich_top3(
                    base_tools=tools,
                    catalog=routing_tools,
                    state_text=state_text,
                    session_id=session_id,
                    turn_id=turn_id,
                )
            elif self.config.hierarchical:
                selected_tools, decision = self._route_hierarchical(
                    tools=routing_tools,
                    state_text=state_text,
                    session_id=session_id,
                    turn_id=turn_id,
                    bridge_tools=bridge_tools,
                )
            else:
                selected_tools, decision = self._route_flat(
                    tools=routing_tools,
                    state_text=state_text,
                    session_id=session_id,
                    turn_id=turn_id,
                )

            # Store in repeat-ask cache
            if cache_key and self.config.cache_repeat_asks and decision.get("decision") in ("execute_tools", "direct_chat", "enrich_top3"):
                self._decision_cache[cache_key] = (selected_tools, decision)
                self._decision_cache.move_to_end(cache_key)
                while len(self._decision_cache) > self.config.cache_size:
                    self._decision_cache.popitem(last=False)

            # In shadow mode, the primary model tool surface MUST NOT BE CHANGED.
            if self.config.is_shadow:
                decision["simulated_selected_tools"] = [extract_tool_name(t) for t in selected_tools]
                self.last_decision = decision
                return tools, decision

            self.last_decision = decision
            return selected_tools, decision


        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000.0
            error_msg = str(exc)
            logger.warning(
                "Laya routing evaluation failed (%s). Using fallback (%s).",
                error_msg,
                self.config.fallback_mode,
            )

            # Enrichment must never strip the standard tool list, even with strip-all fallback.
            fallback_tools = (
                []
                if (
                    self.config.fallback_mode == "strip-all"
                    and not self.config.is_shadow
                    and not self.config.is_enrich_top3
                )
                else tools
            )

            decision = {
                "decision": "fallback",
                "category": None,
                "probability": 0.0,
                "selected_tools": [extract_tool_name(t) for t in fallback_tools],
                "shadow": self.config.is_shadow,
                "error": error_msg,
                "latency_ms": elapsed_ms,
            }
            self.last_decision = decision

            self.audit_logger.log_interaction(
                session_id=session_id,
                turn_id=turn_id,
                mode=self.config.mode,
                decision="fallback",
                selected_tools=[extract_tool_name(t) for t in fallback_tools],
                all_available_tools=[extract_tool_name(t) for t in tools],
                state_snippet=self._safe_state_snippet(state_text),
                latency_ms=elapsed_ms,
                error=error_msg,
            )

            if self.config.is_shadow:
                return tools, decision

            return fallback_tools, decision

    def _route_hierarchical(
        self,
        tools: List[Any],
        state_text: str,
        session_id: str,
        turn_id: str,
        bridge_tools: Optional[List[Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Two-level routing: Level 1 (category) -> Level 2 (tools ranking)."""
        category_groups = self.categorizer.get_categories(tools)

        active_categories = [
            {"id": cat_id, "description": group["description"], "tools": group["tools"]}
            for cat_id, group in category_groups.items()
            if group.get("tools")
        ]

        if not active_categories:
            selected = tools[: self.config.max_tools]
            decision = {
                "decision": "no_active_categories",
                "category": None,
                "probability": 1.0,
                "selected_tools": [extract_tool_name(t) for t in selected],
                "shadow": self.config.is_shadow,
            }
            return selected, decision

        category_criteria = {cat["id"]: cat["description"] for cat in active_categories}

        # Mitigating position bias: formulate balanced questions with clear descriptions
        level1_questions = {
            "tool_decision": {
                "type": "choice",
                "instructions": "Determine whether this request requires tool/command execution or is a pure conversational response.",
                "criteria": {
                    "execute_tools": "Requires executing tools, running terminal commands, reading or editing files, searching web, or external actions",
                    "direct_chat": "Pure conversational chat, greeting, text explanation, or answering general knowledge without any tools",
                },
            },
            "category": {
                "type": "choice",
                "instructions": "If tools are required, which tool category best fits this request?",
                "criteria": category_criteria,
            },
        }

        level1_resp = self.laya.ask(state_text, level1_questions)
        answers = level1_resp.get("answers", {})
        tool_decision_ans = answers.get("tool_decision", {})
        category_ans = answers.get("category", {})

        decision_choice = tool_decision_ans.get("choice")
        cat_choice = category_ans.get("choice")
        cat_probs = category_ans.get("probabilities", {})
        if not cat_probs and cat_choice:
            cat_probs = {cat_choice: float(category_ans.get("confidence") or 0.95)}
        latency_ms = level1_resp.get("latency_ms")

        tool_probs = tool_decision_ans.get("probabilities", {})
        if "direct_chat" in tool_probs:
            direct_prob = float(tool_probs.get("direct_chat", 0.0))
        elif decision_choice == "direct_chat":
            direct_prob = float(tool_decision_ans.get("confidence") or 0.95)
        else:
            direct_prob = 0.0

        if "execute_tools" in tool_probs:
            exec_prob = float(tool_probs.get("execute_tools", 0.0))
        elif decision_choice == "execute_tools":
            exec_prob = float(tool_decision_ans.get("confidence") or 0.95)
        else:
            exec_prob = 0.0

        raw_conf = tool_decision_ans.get("confidence")
        if raw_conf is not None:
            tool_conf = float(raw_conf)
        elif tool_probs:
            tool_conf = max(0.0, direct_prob - exec_prob)
        else:
            tool_conf = 0.90

        # Check for direct chat with safety threshold.
        # Only strip tools if Laya is decisively confident (threshold >= 0.75, conf >= min_confidence).
        # In an agent environment, weak confidence must NEVER strip tools.
        is_confident_direct = (
            (decision_choice == "direct_chat" or decision_choice == NONE_CATEGORY_ID)
            and direct_prob >= self.config.direct_chat_threshold
            and tool_conf >= self.config.min_confidence
        )

        if is_confident_direct:
            decision = {
                "decision": "direct_chat",
                "category": None,
                "probability": float(direct_prob),
                "selected_tools": [],
                "latency_ms": latency_ms,
                "shadow": self.config.is_shadow,
                "error": None,
            }

            self.audit_logger.log_interaction(
                session_id=session_id,
                turn_id=turn_id,
                mode=self.config.mode,
                decision="direct_chat",
                winning_category=None,
                probability=float(direct_prob),
                selected_tools=[],
                all_available_tools=[extract_tool_name(t) for t in tools],
                state_snippet=self._safe_state_snippet(state_text),
                questions=level1_questions,
                raw_answers=answers,
                latency_ms=latency_ms,
            )
            return [], decision

        # Tool category selection
        sorted_categories = sorted(
            active_categories,
            key=lambda c: cat_probs.get(c["id"], 0.0),
            reverse=True,
        )
        best_category = sorted_categories[0]
        cat_prob = float(cat_probs.get(best_category["id"], 0.0))

        # Build candidate pool starting with best category
        candidate_pool: List[Any] = list(best_category["tools"])
        seen_names = {extract_tool_name(t) for t in candidate_pool}

        # Backfill if category has fewer than max_tools
        cat_idx = 1
        while len(candidate_pool) < self.config.max_tools and cat_idx < len(sorted_categories):
            next_cat = sorted_categories[cat_idx]
            cat_idx += 1
            for t in next_cat["tools"]:
                t_name = extract_tool_name(t)
                if t_name not in seen_names:
                    candidate_pool.append(t)
                    seen_names.add(t_name)
                    if len(candidate_pool) >= self.config.max_tools:
                        break

        # Always inject tools from config.always_include into candidate pool
        if self.config.always_include:
            for t in tools:
                t_name = extract_tool_name(t)
                if t_name in self.config.always_include and t_name not in seen_names:
                    candidate_pool.insert(0, t)
                    seen_names.add(t_name)

        # Level 2: If pool <= max_tools, no extra ranking needed
        if len(candidate_pool) <= self.config.max_tools:
            selected_names = [extract_tool_name(t) for t in candidate_pool]
            effective_tools = list(candidate_pool)
            if not self.config.is_shadow and bridge_tools:
                for bt in bridge_tools:
                    bt_name = extract_tool_name(bt)
                    if bt_name not in seen_names:
                        effective_tools.append(bt)
                        seen_names.add(bt_name)

            decision = {
                "decision": "execute_tools",
                "category": best_category["id"],
                "probability": cat_prob,
                "selected_tools": selected_names,
                "latency_ms": latency_ms,
                "shadow": self.config.is_shadow,
                "error": None,
            }
            self.audit_logger.log_interaction(
                session_id=session_id,
                turn_id=turn_id,
                mode=self.config.mode,
                decision="execute_tools",
                winning_category=best_category["id"],
                probability=cat_prob,
                selected_tools=selected_names,
                all_available_tools=[extract_tool_name(t) for t in tools],
                state_snippet=self._safe_state_snippet(state_text),
                questions=level1_questions,
                raw_answers=answers,
                latency_ms=latency_ms,
            )
            return effective_tools, decision

        # Candidate pool > max_tools: rank tools
        top_tools = self.rank_tools(candidate_pool, state_text)
        selected_names = [extract_tool_name(t) for t in top_tools]
        effective_tools = list(top_tools)
        if not self.config.is_shadow and bridge_tools:
            top_names = set(selected_names)
            for bt in bridge_tools:
                bt_name = extract_tool_name(bt)
                if bt_name not in top_names:
                    effective_tools.append(bt)
                    top_names.add(bt_name)

        decision = {
            "decision": "execute_tools",
            "category": best_category["id"],
            "probability": cat_prob,
            "selected_tools": selected_names,
            "latency_ms": latency_ms,
            "shadow": self.config.is_shadow,
            "error": None,
        }

        self.audit_logger.log_interaction(
            session_id=session_id,
            turn_id=turn_id,
            mode=self.config.mode,
            decision="execute_tools",
            winning_category=best_category["id"],
            probability=cat_prob,
            selected_tools=selected_names,
            all_available_tools=[extract_tool_name(t) for t in tools],
            state_snippet=self._safe_state_snippet(state_text),
            questions=level1_questions,
            raw_answers=answers,
            latency_ms=latency_ms,
        )
        return effective_tools, decision

    def _route_enrich_top3(
        self,
        base_tools: List[Any],
        catalog: List[Any],
        state_text: str,
        session_id: str,
        turn_id: str,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Mode "Добавление топ 3": standard tool list + Laya's top 3 tools missing from it.

        Top 3 is found like top 10: with two-level routing Laya first names the category the
        needed tool belongs to, then the top 3 are ranked inside that category; with flat routing
        the top 3 are ranked across the whole catalog. The standard list is never reduced.
        """
        n = ENRICH_TOP_N
        category_id: Optional[str] = "flat"
        cat_prob = 1.0
        latency_ms: Optional[float] = None
        questions: Optional[Dict[str, Any]] = None
        answers: Optional[Dict[str, Any]] = None
        top_tools: List[Any] = []

        category_groups = self.categorizer.get_categories(catalog) if self.config.hierarchical else {}
        active_categories = [
            {"id": cat_id, "description": group["description"], "tools": group["tools"]}
            for cat_id, group in category_groups.items()
            if group.get("tools")
        ]

        if active_categories:
            # Level 1: which category does the needed tool belong to?
            questions = {
                "category": {
                    "type": "choice",
                    "instructions": "Which tool category does the tool needed for this request belong to?",
                    "criteria": {cat["id"]: cat["description"] for cat in active_categories},
                }
            }
            resp = self.laya.ask(state_text, questions)
            answers = resp.get("answers", {})
            latency_ms = resp.get("latency_ms")
            category_ans = answers.get("category", {})
            cat_choice = category_ans.get("choice")
            cat_probs = category_ans.get("probabilities", {})
            if not cat_probs and cat_choice:
                cat_probs = {cat_choice: float(category_ans.get("confidence") or 0.95)}

            sorted_categories = sorted(
                active_categories, key=lambda c: cat_probs.get(c["id"], 0.0), reverse=True
            )
            best_category = sorted_categories[0]
            category_id = best_category["id"]
            cat_prob = float(cat_probs.get(category_id, 0.0))

            # Candidate pool: the winning category, backfilled from the next ones up to n tools.
            pool: List[Any] = list(best_category["tools"])
            seen = {extract_tool_name(t) for t in pool}
            for next_cat in sorted_categories[1:]:
                if len(pool) >= n:
                    break
                for t in next_cat["tools"]:
                    t_name = extract_tool_name(t)
                    if t_name not in seen:
                        pool.append(t)
                        seen.add(t_name)
                        if len(pool) >= n:
                            break

            # Level 2: top n inside the category (no ranking needed when the pool already fits).
            top_tools = pool if len(pool) <= n else self.rank_tools(pool, state_text, limit=n, respect_always=False)
        else:
            # Flat routing (or no categories available): top n across the whole catalog.
            category_id = "flat"
            top_tools = self.rank_tools(catalog, state_text, limit=n, respect_always=False)

        top_tools = top_tools[:n]
        existing = {extract_tool_name(t) for t in base_tools}
        effective_tools = list(base_tools)
        added_names: List[str] = []
        for t in top_tools:
            t_name = extract_tool_name(t)
            if t_name not in existing:
                effective_tools.append(t)
                existing.add(t_name)
                added_names.append(t_name)

        selected_names = [extract_tool_name(t) for t in top_tools]
        decision = {
            "decision": "enrich_top3",
            "category": category_id,
            "probability": cat_prob,
            "selected_tools": selected_names,
            "added_tools": added_names,
            "latency_ms": latency_ms,
            "shadow": False,
            "error": None,
        }
        self.audit_logger.log_interaction(
            session_id=session_id,
            turn_id=turn_id,
            mode=self.config.mode,
            decision="enrich_top3",
            winning_category=category_id,
            probability=cat_prob,
            selected_tools=selected_names,
            all_available_tools=[extract_tool_name(t) for t in catalog],
            state_snippet=self._safe_state_snippet(state_text),
            questions=questions,
            raw_answers=answers,
            latency_ms=latency_ms,
        )
        return effective_tools, decision

    def _route_flat(
        self,
        tools: List[Any],
        state_text: str,
        session_id: str,
        turn_id: str,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Flat single-level ranking without category division."""
        top_tools = self.rank_tools(tools, state_text)
        selected_names = [extract_tool_name(t) for t in top_tools]
        decision = {
            "decision": "execute_tools",
            "category": "flat",
            "probability": 1.0,
            "selected_tools": selected_names,
            "shadow": self.config.is_shadow,
            "error": None,
        }
        self.audit_logger.log_interaction(
            session_id=session_id,
            turn_id=turn_id,
            mode=self.config.mode,
            decision="execute_tools",
            winning_category="flat",
            probability=1.0,
            selected_tools=selected_names,
            all_available_tools=[extract_tool_name(t) for t in tools],
            state_snippet=self._safe_state_snippet(state_text),
        )
        return top_tools, decision

    def rank_tools(
        self,
        tools: List[Any],
        state_text: str,
        limit: Optional[int] = None,
        respect_always: bool = True,
    ) -> List[Any]:
        """Rank candidate tools using Laya choice questions and select top N.

        Args:
            tools: Candidate tool schemas.
            state_text: Cleaned conversation state sent to Laya.
            limit: Number of tools to keep (defaults to ``config.max_tools``; 3 for enrich_top3).
            respect_always: Reserve slots for ``config.always_include`` tools (off for enrich_top3,
                where the standard list already carries them).
        """
        max_n = limit if limit is not None else self.config.max_tools
        always_tools: List[Any] = []
        candidate_tools: List[Any] = []

        for tool in tools:
            name = extract_tool_name(tool)
            if respect_always and name in self.config.always_include:
                always_tools.append(tool)
            else:
                candidate_tools.append(tool)

        slots_remaining = max(0, max_n - len(always_tools))
        if slots_remaining == 0 or not candidate_tools:
            return always_tools

        # Chunk candidate tools into up to 14 options per question
        chunk_size = 14
        chunks = [
            candidate_tools[i : i + chunk_size]
            for i in range(0, len(candidate_tools), chunk_size)
        ]

        tool_scores: Dict[str, float] = {}
        max_questions_per_batch = 8

        for b in range(0, len(chunks), max_questions_per_batch):
            batch_chunks = chunks[b : b + max_questions_per_batch]
            questions: Dict[str, Any] = {}

            for idx, chunk in enumerate(batch_chunks):
                criteria = {}
                for tool in chunk:
                    name = extract_tool_name(tool)
                    desc = (extract_tool_desc(tool) or name).replace("\n", " ")[:120]
                    criteria[name] = desc
                questions[f"batch_{b + idx}"] = {
                    "type": "choice",
                    "instructions": "Which tool is most relevant and necessary to execute this step?",
                    "criteria": criteria,
                }

            resp = self.laya.ask(state_text, questions)
            answers = resp.get("answers", {})

            for idx, chunk in enumerate(batch_chunks):
                ans = answers.get(f"batch_{b + idx}", {})
                probs = ans.get("probabilities", {})
                for tool in chunk:
                    t_name = extract_tool_name(tool)
                    score = float(probs.get(t_name, 0.0))
                    tool_scores[t_name] = score

        ranked_candidates = sorted(
            candidate_tools,
            key=lambda t: tool_scores.get(extract_tool_name(t), 0.0),
            reverse=True,
        )

        selected_candidates = ranked_candidates[:slots_remaining]
        return always_tools + selected_candidates

    def extract_context_state(self, context: Any) -> str:
        """Extract user prompt / state text from various Hermes context representations."""
        if not context:
            return ""

        if isinstance(context, str):
            clean = self._clean_text(context.strip())
            return clean if clean.startswith(("USER:", "SYSTEM:", "ASSISTANT:")) else f"USER: {clean}"

        if isinstance(context, list):
            return self._format_messages(context)

        # Hermes TurnContext or dict with messages/user_message
        if isinstance(context, dict):
            if "user_message" in context and isinstance(context["user_message"], str):
                msg = self._clean_text(context["user_message"].strip())
                if msg:
                    return f"USER: {msg}"
            if "messages" in context and isinstance(context["messages"], list):
                return self._format_messages(context["messages"])
            if "conversation_history" in context and isinstance(context["conversation_history"], list):
                return self._format_messages(context["conversation_history"])

        # Agent object
        if hasattr(context, "session"):
            session = getattr(context, "session")
            if hasattr(session, "derive_messages"):
                return self._format_messages(session.derive_messages())

        return ""

    def _safe_state_snippet(self, state_text: str) -> str:
        """Sanitize state text before saving to audit log if privacy mode is enabled."""
        if self.config.privacy_mode:
            return redact_sensitive(state_text)
        return state_text

    def _clean_text(self, text: str) -> str:
        """Strip plan mode boilerplate, unwrap markdown envelopes, and clean query."""
        if not text:
            return ""
        text = unwrap(text.strip())
        if text.startswith("[/plan"):
            parts = text.split("\n\n")
            for p in reversed(parts):
                p_clean = p.strip()
                if p_clean and not p_clean.startswith(("-", "*", "[/plan")):
                    return p_clean
        return text


    def _format_messages(self, messages: List[Any]) -> str:
        if not messages:
            return ""

        lines: List[str] = []
        slice_msgs = messages[-4:]
        for msg in slice_msgs:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user")).upper()
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    ]
                    content = " ".join(text_parts)
                elif not isinstance(content, str):
                    content = str(content)
                text = self._clean_text(content.strip())
                if text:
                    lines.append(f"{role}: {text[:500]}")
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                role = str(getattr(msg, "role", "user")).upper()
                content = str(getattr(msg, "content", "")).strip()
                text = self._clean_text(content)
                if text:
                    lines.append(f"{role}: {text[:500]}")

        return "\n".join(lines)
