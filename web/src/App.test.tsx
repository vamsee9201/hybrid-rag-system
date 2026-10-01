// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import App, { questionBank } from './App'
import { RagCard } from './RagCard'

afterEach(cleanup)

describe('Hybrid RAG Evaluation UI', () => {
  it('shows the three retrieval strategies and cost disclosure', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Lexical' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Semantic' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Hybrid' })).toBeInTheDocument()
    expect(screen.getByText(/Comparison uses three model answers/)).toBeInTheDocument()
  })

  it('switches to one selected RAG', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Single RAG' }))
    expect(screen.getByLabelText('RAG mode')).toBeInTheDocument()
    expect(screen.getAllByRole('article')).toHaveLength(1)
  })

  it('is single-turn and shows three suggestions from a 30-question bank', () => {
    render(<App />)
    expect(questionBank).toHaveLength(30)
    expect(new Set(questionBank)).toHaveProperty('size', 30)
    expect(screen.queryByRole('button', { name: 'New conversation' })).not.toBeInTheDocument()
    expect(screen.getByText(/Single-turn question answering only/)).toBeInTheDocument()
    expect(within(screen.getByRole('region', { name: 'Suggested questions' })).getAllByRole('button')).toHaveLength(3)
  })

  it('renders answer markdown as formatted content', () => {
    render(<RagCard mode="hybrid" state={{
      status: 'complete',
      text: 'Summary:\n\n* **High-Risk:** Supported claim [GAO-1, p. 2].\n* Second finding',
      passages: [],
    }} />)
    expect(screen.getByRole('list')).toBeInTheDocument()
    expect(screen.getByText('High-Risk:').tagName).toBe('STRONG')
    expect(screen.queryByText(/\*\*High-Risk/)).not.toBeInTheDocument()
  })
})
