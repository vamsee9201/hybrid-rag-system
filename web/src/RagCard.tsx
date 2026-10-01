import { BookOpen, Check, ChevronDown, Clock, Database, TriangleAlert } from 'lucide-react'
import Markdown from 'react-markdown'
import { useState } from 'react'
import remarkGfm from 'remark-gfm'
import type { CardState, RagMode } from './types'

const metadata: Record<RagMode, { title: string; eyebrow: string; description: string }> = {
  bm25: { title: 'Lexical', eyebrow: 'BM25', description: 'Matches exact terms and phrases' },
  dense: { title: 'Semantic', eyebrow: 'VERTEX DENSE', description: 'Matches meaning across different wording' },
  hybrid: { title: 'Hybrid', eyebrow: 'BM25 + DENSE', description: 'Fuses both rankings' },
}

export function RagCard({ mode, state }: { mode: RagMode; state: CardState }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const info = metadata[mode]
  const busy = state.status === 'retrieving' || state.status === 'generating'
  return (
    <article className={`rag-card rag-card--${mode}`} aria-busy={busy}>
      <header className="rag-card__header">
        <div>
          <span className="eyebrow">{info.eyebrow}</span>
          <h2>{info.title}</h2>
          <p>{info.description}</p>
        </div>
        <span className={`status status--${state.status}`}>
          {state.status === 'complete' ? <Check size={14} /> : busy ? <span className="spinner" /> : null}
          {state.status === 'idle' ? 'Ready' : state.status}
        </span>
      </header>

      <div className="answer" aria-live="polite">
        {state.status === 'idle' && <p className="placeholder">Ask a question to see this retriever’s answer.</p>}
        {state.status === 'retrieving' && <p className="placeholder">Searching government publications…</p>}
        {state.status === 'error' && <p className="error"><TriangleAlert size={18} /> {state.error}</p>}
        {state.text && <div className="answer__content">
          <Markdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ children, ...props }) => <a {...props} target="_blank" rel="noreferrer">{children}</a>,
            }}
          >{state.text}</Markdown>
          {busy && <span className="cursor" aria-hidden="true" />}
        </div>}
      </div>

      {(state.passages.length > 0 || state.answer) && (
        <footer>
          <div className="metrics">
            <span><Clock size={14} /> {state.answer ? `${Math.round(state.answer.generation_ms)} ms` : 'Generating'}</span>
            <span><Database size={14} /> {state.passages.length} sources</span>
            {state.answer && <span>${state.answer.usage.estimated_cost_usd.toFixed(4)}</span>}
          </div>
          <button className="sources-toggle" onClick={() => setSourcesOpen(!sourcesOpen)} aria-expanded={sourcesOpen}>
            <BookOpen size={16} /> Evidence <ChevronDown size={16} className={sourcesOpen ? 'rotate' : ''} />
          </button>
          {sourcesOpen && <div className="sources">
            {state.passages.map((passage) => (
              <details key={passage.citation.chunk_id}>
                <summary>
                  <span>{passage.rank}</span>
                  <strong>{passage.citation.title}</strong>
                  <small>p. {passage.citation.page}</small>
                </summary>
                <p>{passage.text}</p>
                {passage.citation.source_url && <a href={passage.citation.source_url} target="_blank" rel="noreferrer">Open GovInfo source ↗</a>}
              </details>
            ))}
          </div>}
        </footer>
      )}
    </article>
  )
}
