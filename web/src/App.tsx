import { ArrowUp, Code2, Columns3 } from 'lucide-react'
import { FormEvent, useMemo, useRef, useState } from 'react'
import { ChatRequestError, streamChat, submitFeedback } from './api'
import { RagCard } from './RagCard'
import type { CardState, Passage, RagAnswer, RagMode } from './types'

const modes: RagMode[] = ['bm25', 'dense', 'hybrid']
const emptyCard = (): CardState => ({ status: 'idle', text: '', passages: [] })
export const questionBank = [
  'What did the GAO report say about improper payments?',
  'Compare the stated budget priorities across two reports.',
  'Which agency was responsible, and on what page is that stated?',
  'What federal programs were identified as high risk, and why?',
  'What were the reported costs of Medicare improper payments?',
  'How did Medicaid spending change during the period discussed?',
  'What findings were reported about SNAP administration?',
  'What cybersecurity weaknesses did the report identify?',
  'What recommendations were made for improving federal oversight?',
  'How did the report describe risks to critical infrastructure?',
  'What funding was allocated for transportation programs?',
  'What did the documents report about veterans health care?',
  'What housing assistance challenges were identified?',
  'What actions were recommended for disaster preparedness?',
  'What evidence was provided about opioid response programs?',
  'How were small businesses affected by the policy discussed?',
  'What workforce shortages were reported by federal agencies?',
  'What did the documents say about public health preparedness?',
  'What environmental risks were identified in the reports?',
  'What energy policy goals were described, and by which agencies?',
  'What support was proposed for agriculture or rural communities?',
  'What procurement problems did the auditors identify?',
  'How did the reports describe federal grant management?',
  'What did the documents report about the national debt?',
  'What tax expenditures or revenue effects were discussed?',
  'What education funding priorities were identified?',
  'What findings concerned Social Security administration?',
  'How did the reports address immigration program operations?',
  'What cross-agency coordination problems were documented?',
  'Which recommendations remained unresolved at the time of the report?',
]

function sampleQuestions(exclude?: string): string[] {
  const shuffled = questionBank.filter((question) => question !== exclude)
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1))
    ;[shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]]
  }
  return shuffled.slice(0, 3)
}

export default function App() {
  const [comparison, setComparison] = useState(true)
  const [singleMode, setSingleMode] = useState<RagMode>('hybrid')
  const [message, setMessage] = useState('')
  const [cards, setCards] = useState<Record<RagMode, CardState>>({ bm25: emptyCard(), dense: emptyCard(), hybrid: emptyCard() })
  const [suggestedQuestions, setSuggestedQuestions] = useState(sampleQuestions)
  const [runId, setRunId] = useState('')
  const [feedback, setFeedback] = useState<RagMode | null>(null)
  const [globalError, setGlobalError] = useState('')
  const controller = useRef<AbortController | null>(null)
  const selectedModes = useMemo(() => comparison ? modes : [singleMode], [comparison, singleMode])
  const busy = selectedModes.some((mode) => ['retrieving', 'generating'].includes(cards[mode].status))

  async function ask(event?: FormEvent, example?: string) {
    event?.preventDefault()
    const question = (example ?? message).trim()
    if (!question || busy) return
    setMessage('')
    setGlobalError('')
    setFeedback(null)
    setRunId('')
    setSuggestedQuestions(sampleQuestions(question))
    const nextCards: Record<RagMode, CardState> = { bm25: emptyCard(), dense: emptyCard(), hybrid: emptyCard() }
    selectedModes.forEach((mode) => { nextCards[mode] = { status: 'retrieving', text: '', passages: [] } })
    setCards(nextCards)
    controller.current = new AbortController()
    try {
      for await (const item of streamChat(question, selectedModes, controller.current.signal)) {
        const mode = item.data.mode as RagMode | undefined
        if (item.event === 'run') setRunId(item.data.run_id as string)
        if (!mode) continue
        if (item.event === 'retrieval') {
          setCards((current) => ({ ...current, [mode]: { ...current[mode], status: 'generating', passages: item.data.passages as Passage[] } }))
        } else if (item.event === 'delta') {
          setCards((current) => ({ ...current, [mode]: { ...current[mode], text: current[mode].text + (item.data.text as string) } }))
        } else if (item.event === 'answer') {
          const answer = item.data as unknown as RagAnswer
          setCards((current) => ({ ...current, [mode]: { status: 'complete', text: answer.answer, passages: answer.passages, answer } }))
        } else if (item.event === 'error') {
          setCards((current) => ({ ...current, [mode]: { ...current[mode], status: 'error', error: item.data.message as string } }))
        }
      }
    } catch (error) {
      if ((error as Error).name === 'AbortError') return
      const message = error instanceof Error ? error.message : 'The request could not be completed.'
      const isRequestLimit = error instanceof ChatRequestError && error.status === 429
      setGlobalError(message)
      setCards((current) => {
        const next = { ...current }
        selectedModes.forEach((mode) => {
          if (next[mode].status !== 'complete') {
            next[mode] = isRequestLimit ? emptyCard() : { ...next[mode], status: 'error', error: message }
          }
        })
        return next
      })
    }
  }

  async function chooseWinner(mode: RagMode) {
    if (!runId || feedback) return
    await submitFeedback(runId, mode)
    setFeedback(mode)
  }

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="/" aria-label="Hybrid RAG Evaluation home">Hybrid RAG Evaluation</a>
      <nav>
        <a href="https://github.com/vamsee9201/hybrid-rag-system" target="_blank" rel="noreferrer"><Code2 size={17} /> Source</a>
      </nav>
    </header>

    <main>
      <section className="hero">
        <span className="kicker">CONTROLLED RETRIEVAL BENCHMARK</span>
        <h1>Hybrid RAG<br /><span>Evaluation</span></h1>
        <p>A controlled comparison of BM25, dense vector retrieval, and reciprocal-rank fusion over 500 U.S. government publications. Each pipeline uses the same corpus, context limit, prompt, and Gemini generator.</p>
      </section>

      <section className="controls" aria-label="Comparison settings">
        <div className="segmented">
          <button className={comparison ? 'active' : ''} onClick={() => setComparison(true)}><Columns3 size={16} /> Compare all</button>
          <button className={!comparison ? 'active' : ''} onClick={() => setComparison(false)}>Single RAG</button>
        </div>
        {!comparison && <select value={singleMode} onChange={(event) => setSingleMode(event.target.value as RagMode)} aria-label="RAG mode">
          <option value="bm25">BM25 lexical</option><option value="dense">Vertex semantic</option><option value="hybrid">Hybrid fusion</option>
        </select>}
      </section>

      <section className={`card-grid ${comparison ? '' : 'card-grid--single'}`}>
        {selectedModes.map((mode) => <RagCard key={mode} mode={mode} state={cards[mode]} />)}
      </section>

      {runId && selectedModes.every((mode) => cards[mode].status === 'complete') && comparison && <section className="feedback">
        <p>Which answer helped most?</p>
        {modes.map((mode) => <button key={mode} disabled={!!feedback} className={feedback === mode ? 'selected' : ''} onClick={() => chooseWinner(mode)}>{mode === 'bm25' ? 'Lexical' : mode === 'dense' ? 'Semantic' : 'Hybrid'}</button>)}
        {feedback && <span>Thanks—recorded anonymously.</span>}
      </section>}

      <form className="composer" onSubmit={(event) => ask(event)}>
        <label htmlFor="question">Ask the documents</label>
        <div><textarea id="question" value={message} maxLength={2000} rows={2} onChange={(event) => setMessage(event.target.value)} onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); ask() }
        }} placeholder="Ask about budgets, reports, hearings, or public policy…" />
        <button type="submit" disabled={!message.trim() || busy} aria-label="Send question"><ArrowUp size={20} /></button></div>
        <small>{message.length}/2,000 · Comparison uses three model answers</small>
      </form>
      <p className="single-turn-note">Single-turn question answering only. Each submission is evaluated independently; no conversation context is retained.</p>
      {globalError && <p className="global-error" role="alert">{globalError}</p>}

      <section className="examples" aria-label="Suggested questions"><span>TRY A QUESTION</span>{suggestedQuestions.map((example) => <button key={example} onClick={() => ask(undefined, example)}>{example}</button>)}</section>
    </main>
  </div>
}
