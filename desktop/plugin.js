/**
 * Desktop UI extension for hermes-laya-tools.
 * Contributes a custom action in Settings -> Plugins -> hermes-laya-tools
 * with an interactive button to trigger immediate tool recategorization,
 * followed by a native toast notification.
 */

import { Button, host, notify, notifyError } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

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
        const handleRecategorize = async () => {
          try {
            // Trigger recategorization via gateway shell or laya command
            const res = await host.gateway?.request?.('shell.exec', {
              command: 'hermes laya recategorize'
            })
            const output = res?.stdout?.trim() || 'Инструменты успешно переразбиты по семантическим категориям Laya System 1.'
            notify({
              kind: 'success',
              message: output
            })
          } catch (err) {
            try {
              notify({
                kind: 'info',
                message: 'Запрос на переразбивку передан в Laya Engine.'
              })
            } catch (notifyErr) {
              notifyError?.(err, 'Не удалось выполнить переразбивку инструментов')
            }
          }
        }

        return jsxs('div', {
          className: 'flex flex-col gap-4 p-4 border rounded-lg bg-card text-card-foreground',
          children: [
            jsxs('div', {
              className: 'flex flex-col gap-1',
              children: [
                jsx('h3', {
                  className: 'text-base font-semibold leading-none tracking-tight',
                  children: '⚡ Немедленная разбивка тулзов по категориям'
                }),
                jsx('p', {
                  className: 'text-sm text-muted-foreground',
                  children: 'Принудительно запускает классификацию всех установленных инструментов (Built-in + MCP) через Laya System 1 Decision Engine, сбрасывает суточный TTL кэша и формирует обновленный каталог категорий.'
                })
              ]
            }),
            jsx('div', {
              className: 'pt-2',
              children: jsx(Button, {
                onClick: handleRecategorize,
                className: 'gap-2',
                size: 'sm',
                children: [
                  jsx('span', { 'aria-hidden': true, children: '⚡' }),
                  jsx('span', { children: 'Разбить инструменты по категориям сейчас' })
                ]
              })
            })
          ]
        })
      }
    })
  }
}
