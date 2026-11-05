/**
 * Desktop UI extension for hermes-laya-tools.
 *
 * Contributes the settings page Settings -> Plugins -> hermes-laya-tools with an action BUTTON
 * "Переразбить категории сейчас" (the manifest `config_schema` has no button type, so the action lives
 * here instead of being a switch in the schema form).
 *
 * The button runs the plugin's own `/laya recategorize` command in the focused chat session through the
 * gateway (`slash.exec`), so the tools are recategorized immediately and the summary is shown as a toast.
 */

import { Button, ListRow, host, notify, notifyError } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

const BUTTON_LABEL = '⚡ Переразбить сейчас'
const RECATEGORIZE_COMMAND = 'laya recategorize'

/** Run `/laya recategorize` in the focused session; toast the outcome. Resolves true on success. */
async function recategorizeNow() {
  const sessionId = host.state.focusedSessionId.get() ?? host.state.activeSessionId.get()

  if (!sessionId) {
    notify({
      kind: 'info',
      message: 'Откройте любой чат и нажмите кнопку ещё раз: переразбивка выполняется командой /laya recategorize в рабочей сессии.'
    })

    return false
  }

  try {
    const reply = await host.request('slash.exec', { session_id: sessionId, command: RECATEGORIZE_COMMAND })

    notify({
      kind: 'success',
      message: reply?.output || reply?.notice || 'Инструменты переразбиты по категориям Laya System 1.'
    })

    return true
  } catch (err) {
    notifyError(err, 'Не удалось переразбить инструменты по категориям')

    return false
  }
}

export default {
  id: 'hermes-laya-tools',
  name: 'Hermes Laya Tools',

  register(ctx) {
    ctx.registerSettingsPage?.({
      id: 'laya-tools-actions',
      title: 'Категоризация инструментов',
      icon: 'sparkle',
      order: 10,
      render: function LayaActionsPage() {
        return jsx(ListRow, {
          title: 'Переразбить категории сейчас',
          description:
            'Принудительно запускает немедленную переразбивку всех доступных инструментов (Built-in + MCP) по категориям через Laya и обновляет кэш категорий.',
          action: jsx(Button, {
            onClick: recategorizeNow,
            size: 'sm',
            type: 'button',
            children: BUTTON_LABEL
          })
        })
      }
    })
  }
}
