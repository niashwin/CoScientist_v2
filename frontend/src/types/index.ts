// Core data types
export interface Hypothesis {
  id: string;
  content: string;
  summary: string;
  scores: {
    novelty: number;
    feasibility: number;
    impact: number;
    testability: number;
    composite: number;
  };
  streamingContent?: string;  // For partial content during streaming
  isStreaming: boolean;
  createdByAgent: string;
  supportingLiterature: Paper[];
  reviews: Review[];
  rank?: number;
  generation: number;
  parentId?: string;
  evolutionMethod?: string;
  createdAt: string;
  experimentalProtocol?: ExperimentalProtocol;
}

export interface Review {
  id: string;
  hypothesisId: string;
  type: 'initial' | 'full' | 'deep_verification' | 'observation' | 'simulation';
  content: string;
  critiques: Critique[];
  suggestions: string[];
  scores: {
    correctness: number;
    novelty: number;
    quality: number;
  };
  noveltyAssessment: 'known' | 'incremental' | 'novel' | 'breakthrough';
  safetyConcerns: string[];
  isStreaming: boolean;
  createdAt: string;
  createdByAgent: string;
  literatureCited: Paper[];
}

export interface Critique {
  aspect: string;
  severity: 'low' | 'medium' | 'high';
  description: string;
  suggestion?: string;
}

export interface Paper {
  title: string;
  authors: string[];
  year: number;
  doi?: string;
  abstract?: string;
  relevance: number;
  citationCount?: number;
  url?: string;
}

export interface ExperimentalProtocol {
  title: string;
  objective: string;
  materials: string[];
  methods: string[];
  expectedOutcomes: string[];
  timeline: string;
  budget?: string;
  risksAndMitigation: string[];
}

// Agent and system state types
export interface AgentState {
  name: string;
  status: 'idle' | 'thinking' | 'streaming' | 'complete' | 'error';
  currentOutput: string;
  message?: string;
  progress?: number;
}

export interface ResearchSession {
  id: string;
  goal: string;
  preferences?: ResearchPreferences;
  status: 'initializing' | 'active' | 'paused' | 'completed' | 'error';
  startedAt: string;
  completedAt?: string;
  agents: Record<string, AgentState>;
  hypotheses: Hypothesis[];
  currentStage: string;
  currentAgent?: string;
  currentIteration: number;
  progress: number;
  error?: SystemError;
  stats: SessionStats;
}

export interface ResearchPreferences {
  focusArea?: string;
  experimentalConstraints?: string[];
  noveltyThreshold?: number;
  maxHypotheses?: number;
  maxIterations?: number;
  includeExperimentalProtocols?: boolean;
  prioritizeTestability?: boolean;
  excludeKnownMechanisms?: boolean;
}

export interface SessionStats {
  hypothesesGenerated: number;
  reviewsCompleted: number;
  tournamentsRun: number;
  literaturePapersReviewed: number;
  executionTimeMs: number;
  agentActivations: Record<string, number>;
}

export interface SystemError {
  code: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  suggestions: string[];
  retryAfter?: number;
  context?: Record<string, any>;
}

// UI state types
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';
export type AppMode = 'simple' | 'advanced';
export type ViewMode = 'overview' | 'detailed' | 'comparison';

export interface UIState {
  mode: AppMode;
  viewMode: ViewMode;
  selectedHypothesisId: string | null;
  expandedSections: Set<string>;
  streamingAgents: string[];
  showDebugInfo: boolean;
  sidebarCollapsed: boolean;
  autoScrollEnabled: boolean;
}

// WebSocket message types
export interface StreamChunk {
  type: 'stream_chunk';
  agent: string;
  content: string;
  chunkId: number;
  timestamp: number;
}

export interface StreamComplete {
  type: 'stream_complete';
  agent: string;
  fullResponse: string;
  timestamp: number;
}

export interface AgentThinking {
  type: 'thinking_start' | 'thinking_chunk' | 'thinking_complete';
  agent: string;
  message?: string;
  content?: string;
  timestamp: number;
}

export interface HypothesisUpdate {
  type: 'hypothesis_generated' | 'hypothesis_reviewed' | 'hypothesis_ranked' | 'hypothesis_evolved' | 'hypothesis_updated';
  hypothesis: Hypothesis;
  sessionId: string;
}

export interface ProgressUpdate {
  type: 'progress';
  stage: string;
  progress: number;
  message?: string;
  sessionId: string;
}

export interface StageUpdate {
  type: 'stage_update';
  stage: string;
  message: string;
  sessionId: string;
}

export interface ErrorMessage {
  type: 'error';
  error: SystemError;
  sessionId: string;
}

export interface SessionComplete {
  type: 'session_complete';
  sessionId: string;
  summary: SessionSummary;
}

export interface LiteratureSearchUpdate {
  type: 'literature_search_update';
  stage: 'searching' | 'analyzing' | 'complete';
  papers: Paper[];
  query: string;
  message: string;
  sessionId: string;
}

export interface SessionSummary {
  totalHypotheses: number;
  topHypothesis: Hypothesis;
  keyInsights: string[];
  suggestedNextSteps: string[];
  executionTime: number;
  qualityMetrics: {
    averageNoveltyScore: number;
    averageFeasibilityScore: number;
    averageImpactScore: number;
    averageTestabilityScore: number;
  };
}

// API request/response types
export interface StartResearchRequest {
  goal: string;
  preferences?: ResearchPreferences;
  mode: AppMode;
}

export interface FeedbackRequest {
  hypothesisId: string;
  feedbackType: 'review' | 'correction' | 'suggestion' | 'rating';
  content: string;
  ratings?: {
    scientificAccuracy: number;
    novelty: number;
    feasibility: number;
    impact: number;
  };
}

export interface ExportRequest {
  sessionId: string;
  format: 'json' | 'pdf' | 'csv';
  includeReviews: boolean;
  includeLiterature: boolean;
}

// Utility types
export type WebSocketMessage = 
  | StreamChunk 
  | StreamComplete 
  | AgentThinking 
  | HypothesisUpdate 
  | ProgressUpdate 
  | StageUpdate 
  | ErrorMessage 
  | SessionComplete
  | LiteratureSearchUpdate;

export interface NotificationMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: number;
  autoClose?: boolean;
  duration?: number;
}

// Component prop types
export interface HypothesisCardProps {
  hypothesis: Hypothesis;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onExpand?: (id: string) => void;
  showReviews?: boolean;
  compact?: boolean;
  showSeedButton?: boolean;
  onSeedNewProcess?: (hypothesis: Hypothesis) => void;
}

export interface StreamingDisplayProps {
  agent: AgentState;
  isActive: boolean;
  showTimestamp?: boolean;
  maxHeight?: string;
  defaultExpanded?: boolean;
}

export interface ProgressBarProps {
  stage: string;
  progress: number;
  showDetails?: boolean;
  animated?: boolean;
}

export interface ConnectionIndicatorProps {
  status: ConnectionStatus;
  showDetails?: boolean;
  className?: string;
} 