"""Tool Router powered by Laya System 1 Decision Engine.

Implements hierarchical two-level routing and Top-10 selection:
1. Daily tool categorization (24h TTL).
2. Level 1: Context gatekeeping (direct_chat vs execute_tools) and Category selection.
   - If direct_chat: strips tools (tools = []).
3. Level 2: Top-N tool ranking for the chosen category.
4. Shadow mode: performs full Laya query, response processing, and logging,
   but leaves the primary model toolset completely unchanged.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .categorizer import ToolCategorizer, extract_tool_desc, extract_tool_name
    from .client import LayaClient
    from .config import LayaConfig
    from .logger import LayaAuditLogger
except ImportError:
    from categorizer import ToolCategorizer, extract_tool_desc, extract_tool_name
    from client import LayaClient
    from config import LayaConfig
    from logger import LayaAuditLogger

logger = logging.getLogger("hermes.plugin.laya.router")

NONE_CATEGORY_ID = "none"


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

    def route_tools(
        self,
        tools: List[Any],
        context: Any,
        session_id: str = "",
        turn_id: str = "",
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Route and filter tools for the upcoming model call.

        Args:
            tools: Available tool schemas from Hermes agent.
            context: Prompt string, messages list, or TurnContext.
            session_id: Current Hermes session identifier.
            turn_id: Current Hermes turn identifier.

        Returns:
            Tuple of (effective_tools, decision_info).
            In shadow mode, effective_tools is always the original tools list.
        """
        if not self.config.enabled or not tools:
            decision = {
                "decision": "disabled" if not self.config.enabled else "no_tools",
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

        start_time = time.time()
        try:
            if self.config.hierarchical:
                selected_tools, decision = self._route_hierarchical(
                    tools=tools,
                    state_text=state_text,
                    session_id=session_id,
                    turn_id=turn_id,
                )
            else:
                selected_tools, decision = self._route_flat(
                    tools=tools,
                    state_text=state_text,
                    session_id=session_id,
                    turn_id=turn_id,
                )

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

            fallback_tools = (
                [] if (self.config.fallback_mode == "strip-all" and not self.config.is_shadow) else tools
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
                state_snippet=state_text,
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

        level1_questions = {
            "tool_decision": {
                "type": "choice",
                "instructions": "Does this user input require executing tools/commands, or is it a direct conversational reply?",
                "criteria": {
                    "direct_chat": "Direct conversational chat, greeting, text explanation, answering general knowledge without any tools",
                    "execute_tools": "Requires executing tools, commands, reading files, searching web, or external actions",
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
        latency_ms = level1_resp.get("latency_ms")

        # Check for direct chat
        if (
            decision_choice == "direct_chat"
            or cat_choice == NONE_CATEGORY_ID
            or decision_choice == NONE_CATEGORY_ID
        ):
            direct_prob = (
                tool_decision_ans.get("probabilities", {}).get("direct_chat")
                or tool_decision_ans.get("confidence")
                or 0.95
            )
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
                state_snippet=state_text,
                questions=level1_questions,
                raw_answers=answers,
                latency_ms=latency_ms,
            )
            return [], decision

        # Tool category won
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

        # Level 2: If pool <= max_tools, no extra ranking needed
        if len(candidate_pool) <= self.config.max_tools:
            selected_names = [extract_tool_name(t) for t in candidate_pool]
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
                state_snippet=state_text,
                questions=level1_questions,
                raw_answers=answers,
                latency_ms=latency_ms,
            )
            return candidate_pool, decision

        # Candidate pool > max_tools: rank tools
        top_tools = self.rank_tools(candidate_pool, state_text)
        selected_names = [extract_tool_name(t) for t in top_tools]

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
            state_snippet=state_text,
            questions=level1_questions,
            raw_answers=answers,
            latency_ms=latency_ms,
        )
        return top_tools, decision

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
            state_snippet=state_text,
        )
        return top_tools, decision

    def rank_tools(self, tools: List[Any], state_text: str) -> List[Any]:
        """Rank candidate tools using Laya choice questions and select top N."""
        always_tools: List[Any] = []
        candidate_tools: List[Any] = []

        for tool in tools:
            name = extract_tool_name(tool)
            if name in self.config.always_include:
                always_tools.append(tool)
            else:
                candidate_tools.append(tool)

        slots_remaining = max(0, self.config.max_tools - len(always_tools))
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
            trimmed = context.strip()
            return trimmed if trimmed.startswith(("USER:", "SYSTEM:", "ASSISTANT:")) else f"USER: {trimmed}"

        if isinstance(context, list):
            return self._format_messages(context)

        # Hermes TurnContext or dict with messages/user_message
        if isinstance(context, dict):
            if "user_message" in context and isinstance(context["user_message"], str):
                msg = context["user_message"].strip()
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
                text = content.strip()
                if text:
                    lines.append(f"{role}: {text[:500]}")
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                role = str(getattr(msg, "role", "user")).upper()
                content = str(getattr(msg, "content", "")).strip()
                if content:
                    lines.append(f"{role}: {content[:500]}")

        return "\n".join(lines)
