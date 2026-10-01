import type { RagMode } from './types'

export interface StreamEvent { event: string; data: Record<string, unknown> }

export class ChatRequestError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'ChatRequestError'
  }
}

export function friendlyChatError(status: number, detail?: string): string {
  const normalized = (detail || '').toLowerCase()
  if (status === 429 && normalized.includes('per-minute')) {
    return 'Request limit reached. Please wait one minute, then try again.'
  }
  if (status === 429 && normalized.includes('daily model budget')) {
    return 'Daily demo budget reached. New questions will be available after the quota resets at midnight UTC.'
  }
  if (status === 429) {
    return 'Daily demo limit reached. This browser’s answer allowance resets at midnight UTC.'
  }
  return detail || `The request could not be completed (${status}).`
}

export async function* streamChat(
  message: string,
  modes: RagMode[],
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, modes }),
    signal,
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new ChatRequestError(friendlyChatError(response.status, payload.detail), response.status)
  }
  if (!response.body) throw new Error('Streaming is unavailable in this browser')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const frames = buffer.split('\n\n')
    buffer = frames.pop() || ''
    for (const frame of frames) {
      let event = 'message'
      let data = '{}'
      for (const line of frame.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7)
        if (line.startsWith('data: ')) data = line.slice(6)
      }
      yield { event, data: JSON.parse(data) }
    }
    if (done) break
  }
}

export async function submitFeedback(runId: string, preferredMode: RagMode) {
  const response = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId, preferred_mode: preferredMode, reason_tags: [] }),
  })
  if (!response.ok) throw new Error('Feedback could not be saved')
}
