export type RagMode = 'bm25' | 'dense' | 'hybrid'

export interface Citation { document_id: string; title: string; page: number; source_url?: string; chunk_id: string }
export interface Passage {
  citation: Citation
  text: string
  rank: number
  score?: number
  bm25_rank?: number
  dense_rank?: number
}
export interface Usage { input_tokens: number; output_tokens: number; total_tokens: number; estimated_cost_usd: number }
export interface RagAnswer {
  mode: RagMode
  status: 'ok' | 'error'
  answer: string
  citations: Citation[]
  passages: Passage[]
  retrieval_ms: number
  generation_ms: number
  usage: Usage
  model: string
  index_version: string
  error?: string
}
export interface CardState {
  status: 'idle' | 'retrieving' | 'generating' | 'complete' | 'error'
  text: string
  passages: Passage[]
  answer?: RagAnswer
  error?: string
}
