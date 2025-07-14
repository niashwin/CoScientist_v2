import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { 
  ResearchSession, 
  Hypothesis, 
  Review, 
  UIState, 
  ConnectionStatus, 
  AppMode, 
  ViewMode,
  NotificationMessage,
  SystemError,
  StartResearchRequest,
  Paper
} from '../types';
import { ResilientWebSocketClient } from '../services/WebSocketClient';

interface CoScientistStore {
  // WebSocket connection
  wsClient: ResilientWebSocketClient | null;
  connectionStatus: ConnectionStatus;
  
  // Research session
  currentSession: ResearchSession | null;
  sessionHistory: ResearchSession[];
  
  // Literature search state
  literatureSearchState: {
    isSearching: boolean;
    query: string;
    papers: Paper[];
    searchStage: 'searching' | 'analyzing' | 'complete' | 'idle';
    message: string;
  };
  
  // UI state
  ui: UIState;
  
  // Notifications
  notifications: NotificationMessage[];
  
  // Actions
  initializeWebSocket: (url: string) => void;
  startResearch: (request: StartResearchRequest) => void;
  pauseResearch: () => void;
  resumeResearch: () => void;
  stopResearch: () => void;
  stopResearchConfirmed: () => void;
  restartResearch: () => void;
  seedNewProcess: (hypothesis: Hypothesis) => void;
  
  // Streaming updates
  handleStreamChunk: (agentName: string, chunk: string) => void;
  handleHypothesisGenerated: (hypothesis: Hypothesis) => void;
  handleHypothesisUpdated: (hypothesis: Hypothesis) => void;
  handleHypothesesFiltered: (hypotheses: Hypothesis[]) => void;
  handleReviewCompleted: (hypothesisId: string, review: Review) => void;
  handleProgressUpdate: (stage: string, progress: number, message?: string) => void;
  handleStageUpdate: (stage: string, message: string) => void;
  handleError: (error: SystemError) => void;
  handleSessionComplete: (sessionId: string) => void;
  handleLiteratureSearchUpdate: (query: string, stage: 'searching' | 'analyzing' | 'complete', papers: Paper[], message: string) => void;
  
  // UI interactions
  setMode: (mode: AppMode) => void;
  setViewMode: (viewMode: ViewMode) => void;
  selectHypothesis: (id: string | null) => void;
  toggleSection: (sectionId: string) => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleDebugInfo: () => void;
  toggleAutoScroll: () => void;
  
  // Notifications
  addNotification: (notification: Omit<NotificationMessage, 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  
  // Session management
  saveSession: (session: ResearchSession) => void;
  loadSession: (sessionId: string) => void;
  exportResults: (format: 'json' | 'pdf' | 'csv') => void;
  
  // Feedback
  submitFeedback: (hypothesisId: string, feedback: any) => void;
}

const initialUIState: UIState = {
  mode: 'simple',
  viewMode: 'overview',
  selectedHypothesisId: null,
  expandedSections: new Set(),
  streamingAgents: [],
  showDebugInfo: false,
  sidebarCollapsed: false,
  autoScrollEnabled: false,
};

export const useCoScientistStore = create<CoScientistStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Initial state
        wsClient: null,
        connectionStatus: 'disconnected',
        currentSession: null,
        sessionHistory: [],
        literatureSearchState: {
          isSearching: false,
          query: '',
          papers: [],
          searchStage: 'idle',
          message: ''
        },
        ui: initialUIState,
        notifications: [],
        
        // Initialize WebSocket connection
        initializeWebSocket: (url: string) => {
          // Set up message handlers first
          const messageHandlers = {
            'stream_chunk': (message: any) => {
              try {
                console.log('Received stream_chunk:', message);
                get().handleStreamChunk(message.agent, message.content);
              } catch (error) {
                console.error('Error handling stream_chunk:', error);
              }
            },
            'hypothesis_generated': (message: any) => {
              try {
                console.log('Received hypothesis_generated:', message);
                get().handleHypothesisGenerated(message.hypothesis);
              } catch (error) {
                console.error('Error handling hypothesis_generated:', error);
              }
            },
            'hypothesis_updated': (message: any) => {
              try {
                console.log('Received hypothesis_updated:', message);
                get().handleHypothesisUpdated(message.hypothesis);
              } catch (error) {
                console.error('Error handling hypothesis_updated:', error);
              }
            },
            'hypotheses_filtered': (message: any) => {
              try {
                console.log('Received hypotheses_filtered:', message);
                get().handleHypothesesFiltered(message.hypotheses);
              } catch (error) {
                console.error('Error handling hypotheses_filtered:', error);
              }
            },
            'review_completed': (message: any) => {
              try {
                console.log('Received review_completed:', message);
                get().handleReviewCompleted(message.hypothesis_id, message.review);
              } catch (error) {
                console.error('Error handling review_completed:', error);
              }
            },
            'progress': (message: any) => {
              try {
                console.log('Received progress:', message);
                get().handleProgressUpdate(message.stage, message.progress, message.message);
              } catch (error) {
                console.error('Error handling progress:', error);
              }
            },
            'stage_update': (message: any) => {
              try {
                console.log('Received stage_update:', message);
                get().handleStageUpdate(message.stage, message.message);
              } catch (error) {
                console.error('Error handling stage_update:', error);
              }
            },
            'error': (message: any) => {
              try {
                console.log('Received error:', message);
                get().handleError(message.error);
              } catch (error) {
                console.error('Error handling error message:', error);
              }
            },
            'session_complete': (message: any) => {
              try {
                console.log('Received session_complete:', message);
                get().handleSessionComplete(message.sessionId);
              } catch (error) {
                console.error('Error handling session_complete:', error);
              }
            },
            'research_complete': (message: any) => {
              try {
                console.log('Received research_complete:', message);
                get().handleSessionComplete(message.session_id);
                get().addNotification({
                  type: 'success',
                  title: 'Research Complete',
                  message: message.message || 'Research pipeline completed successfully',
                  autoClose: true,
                  duration: 5000
                });
              } catch (error) {
                console.error('Error handling research_complete:', error);
              }
            },
            'session_terminated': (message: any) => {
              try {
                console.log('Received session_terminated:', message);
                get().handleSessionComplete(message.sessionId);
                get().addNotification({
                  type: 'warning',
                  title: 'Research Terminated',
                  message: message.message || 'Research session was terminated',
                  autoClose: true,
                  duration: 5000
                });
              } catch (error) {
                console.error('Error handling session_terminated:', error);
              }
            },
            'session_stopped': (message: any) => {
              try {
                console.log('Received session_stopped:', message);
                get().addNotification({
                  type: 'info',
                  title: 'Stop Requested',
                  message: message.message || 'Research termination requested',
                  autoClose: true,
                  duration: 3000
                });
              } catch (error) {
                console.error('Error handling session_stopped:', error);
              }
            },
            'session_started': (message: any) => {
              try {
                console.log('Received session_started:', message);
                get().addNotification({
                  type: 'info',
                  title: 'Session Started',
                  message: message.message || 'Research session has started',
                  autoClose: true,
                  duration: 3000
                });
              } catch (error) {
                console.error('Error handling session_started:', error);
              }
            },
            'agent_start': (message: any) => {
              try {
                console.log('Received agent_start:', message);
                set((state) => {
                  if (!state.currentSession) return;
                  
                  // Ensure agents object exists
                  if (!state.currentSession.agents) {
                    state.currentSession.agents = {};
                  }
                  
                  // Create or update agent state
                  if (!state.currentSession.agents[message.agent]) {
                    state.currentSession.agents[message.agent] = {
                      name: message.agent,
                      status: 'thinking',
                      currentOutput: '',
                      message: message.message || `${message.agent} is starting...`
                    };
                  } else {
                    state.currentSession.agents[message.agent].status = 'thinking';
                    state.currentSession.agents[message.agent].message = message.message || `${message.agent} is starting...`;
                  }
                });
              } catch (error) {
                console.error('Error handling agent_start:', error);
              }
            },
            'agent_complete': (message: any) => {
              try {
                console.log('Received agent_complete:', message);
                set((state) => {
                  if (!state.currentSession) return;
                  
                  const agent = state.currentSession.agents[message.agent];
                  if (agent) {
                    agent.status = 'complete';
                    agent.message = message.message || `${message.agent} completed`;
                  }
                  
                  // Remove from streaming set
                  const agentIndex = state.ui.streamingAgents.indexOf(message.agent);
                  if (agentIndex > -1) {
                    state.ui.streamingAgents.splice(agentIndex, 1);
                  }
                });
              } catch (error) {
                console.error('Error handling agent_complete:', error);
              }
            },
            'agent_response': (message: any) => {
              try {
                console.log('Received agent_response:', message);
                // This is used internally by the backend for processing, 
                // but we can log it for debugging
              } catch (error) {
                console.error('Error handling agent_response:', error);
              }
            },
            'literature_search_update': (message: any) => {
              try {
                console.log('Received literature_search_update:', message);
                get().handleLiteratureSearchUpdate(
                  message.query,
                  message.stage,
                  message.papers || [],
                  message.message
                );
              } catch (error) {
                console.error('Error handling literature_search_update:', error);
              }
            }
          };

          const client = new ResilientWebSocketClient(url, {
            messageHandlers,
            onConnect: () => {
              set((state) => {
                state.connectionStatus = 'connected';
              });
              get().addNotification({
                type: 'success',
                title: 'Connected',
                message: 'Successfully connected to AI Co-Scientist system',
                autoClose: true,
                duration: 3000
              });
            },
            onDisconnect: () => {
              set((state) => {
                state.connectionStatus = 'disconnected';
              });
              get().addNotification({
                type: 'warning',
                title: 'Disconnected',
                message: 'Connection to server lost. Attempting to reconnect...',
                autoClose: true,
                duration: 5000
              });
            },
            onReconnect: () => {
              set((state) => {
                state.connectionStatus = 'connected';
              });
              get().addNotification({
                type: 'info',
                title: 'Reconnected',
                message: 'Successfully reconnected to server',
                autoClose: true,
                duration: 3000
              });
            },
            onError: (_error) => {
              get().addNotification({
                type: 'error',
                title: 'Connection Error',
                message: 'Failed to connect to server. Please check your connection.',
                autoClose: false
              });
            }
          });
          
          set((state) => {
            state.wsClient = client;
            state.connectionStatus = 'connecting';
          });
        },
        
        // Start new research
        startResearch: (request: StartResearchRequest) => {
          const sessionId = crypto.randomUUID();
          const session: ResearchSession = {
            id: sessionId,
            goal: request.goal,
            preferences: request.preferences,
            status: 'initializing',
            startedAt: new Date().toISOString(),
            agents: {
              Supervisor: { name: 'Supervisor', status: 'idle', currentOutput: '' },
              Generation: { name: 'Generation', status: 'idle', currentOutput: '' },
              Reflection: { name: 'Reflection', status: 'idle', currentOutput: '' },
              Ranking: { name: 'Ranking', status: 'idle', currentOutput: '' },
              Evolution: { name: 'Evolution', status: 'idle', currentOutput: '' },
              MetaReview: { name: 'MetaReview', status: 'idle', currentOutput: '' }
            },
            hypotheses: [],
            currentStage: 'initialization',
            currentAgent: undefined,
            currentIteration: 1,
            progress: 0,
            stats: {
              hypothesesGenerated: 0,
              reviewsCompleted: 0,
              tournamentsRun: 0,
              literaturePapersReviewed: 0,
              executionTimeMs: 0,
              agentActivations: {}
            }
          };
          
          set((state) => {
            state.currentSession = session;
            state.ui.mode = request.mode;
            
            // Reset literature search state for new session
            state.literatureSearchState = {
              isSearching: false,
              query: '',
              papers: [],
              searchStage: 'idle',
              message: ''
            };
          });
          
          // Start runtime counter
          const runtimeInterval = setInterval(() => {
            set((state) => {
              if (state.currentSession && (state.currentSession.status === 'active' || state.currentSession.status === 'initializing')) {
                const elapsed = Date.now() - new Date(state.currentSession.startedAt).getTime();
                state.currentSession.stats.executionTimeMs = elapsed;
              } else {
                // Clear interval if session is no longer active
                clearInterval(runtimeInterval);
              }
            });
          }, 1000);
          
          // Send to WebSocket
          get().wsClient?.send({
            type: 'start_research',
            goal: request.goal,
            session_id: sessionId,
            preferences: request.preferences,
            mode: request.mode
          });
          
          get().addNotification({
            type: 'info',
            title: 'Research Started',
            message: `Starting research: "${request.goal}"`,
            autoClose: true,
            duration: 5000
          });
        },
        
        // Handle streaming chunks
        handleStreamChunk: (agentName: string, chunk: string) => {
          console.log('handleStreamChunk called with:', { agentName, chunk });
          set((state) => {
            if (!state.currentSession) {
              console.log('No current session, skipping stream chunk');
              return;
            }
            
            console.log('Processing stream chunk for agent:', agentName);
            console.log('Current session agents:', state.currentSession.agents);
            
            // Ensure agents object exists
            if (!state.currentSession.agents) {
              console.log('Creating agents object');
              state.currentSession.agents = {};
            }
            
            // Create agent state if it doesn't exist
            if (!state.currentSession.agents[agentName]) {
              console.log('Creating agent state for:', agentName);
              state.currentSession.agents[agentName] = {
                name: agentName,
                status: 'streaming',
                currentOutput: '',
                message: `${agentName} is generating insights...`,
                progress: undefined
              };
            }
            
            // Update agent state
            const agent = state.currentSession.agents[agentName];
            const previousOutput = agent.currentOutput;
            agent.status = 'streaming';
            agent.currentOutput += chunk;
            agent.message = `${agentName} is generating insights...`;
            
            console.log('Updated agent state:', {
              name: agentName,
              previousLength: previousOutput.length,
              newLength: agent.currentOutput.length,
              chunkLength: chunk.length,
              status: agent.status
            });
            
            // Add to streaming agents array if not already present
            if (!state.ui.streamingAgents.includes(agentName)) {
              state.ui.streamingAgents.push(agentName);
            }
            console.log('Streaming agents set:', state.ui.streamingAgents);
            
            // If streaming hypothesis content
            const streamingHypothesis = state.currentSession.hypotheses.find(
              h => h.isStreaming && h.createdByAgent === agentName
            );
            
            if (streamingHypothesis) {
              streamingHypothesis.streamingContent = 
                (streamingHypothesis.streamingContent || '') + chunk;
            }
          });
        },
        
        // Handle completed hypothesis
        handleHypothesisGenerated: (hypothesis: Hypothesis) => {
          console.log('handleHypothesisGenerated called with:', hypothesis);
          set((state) => {
            if (!state.currentSession) {
              console.log('No current session, skipping hypothesis generation');
              return;
            }
            
            console.log('Adding hypothesis to session:', hypothesis.id);
            
            // Add hypothesis to session
            state.currentSession.hypotheses.push({
              ...hypothesis,
              isStreaming: false,
              reviews: []
            });
            
            console.log('Total hypotheses now:', state.currentSession.hypotheses.length);
            
            // Update stats
            state.currentSession.stats.hypothesesGenerated++;
            
            // Update agent status
            const agent = state.currentSession.agents[hypothesis.createdByAgent];
            if (agent) {
              agent.status = 'complete';
              console.log('Updated agent status to complete:', hypothesis.createdByAgent);
            }
            
            // Remove from streaming set
            const agentIndex = state.ui.streamingAgents.indexOf(hypothesis.createdByAgent);
            if (agentIndex > -1) {
              state.ui.streamingAgents.splice(agentIndex, 1);
            }
          });
          
          get().addNotification({
            type: 'success',
            title: 'Hypothesis Generated',
            message: `New hypothesis created by ${hypothesis.createdByAgent} agent`,
            autoClose: true,
            duration: 4000
          });
        },
        
        // Handle updated hypothesis (e.g., after scoring by ranking agent)
        handleHypothesisUpdated: (hypothesis: Hypothesis) => {
          console.log('handleHypothesisUpdated called with:', hypothesis);
          set((state) => {
            if (!state.currentSession) {
              console.log('No current session, skipping hypothesis update');
              return;
            }
            
            // Find and update the existing hypothesis
            const existingIndex = state.currentSession.hypotheses.findIndex(h => h.id === hypothesis.id);
            if (existingIndex !== -1) {
              console.log('Updating existing hypothesis:', hypothesis.id);
              // Update the hypothesis while preserving other properties
              state.currentSession.hypotheses[existingIndex] = {
                ...state.currentSession.hypotheses[existingIndex],
                ...hypothesis,
                isStreaming: false
              };
              console.log('Updated hypothesis scores:', state.currentSession.hypotheses[existingIndex].scores);
            } else {
              console.log('Hypothesis not found for update, adding as new:', hypothesis.id);
              // If not found, add as new (fallback)
              state.currentSession.hypotheses.push({
                ...hypothesis,
                isStreaming: false,
                reviews: []
              });
            }
          });
          
          get().addNotification({
            type: 'info',
            title: 'Hypothesis Updated',
            message: `Hypothesis scores updated by ranking agent`,
            autoClose: true,
            duration: 3000
          });
        },
        
        // Handle filtered hypotheses list (after ranking agent filtering)
        handleHypothesesFiltered: (hypotheses: Hypothesis[]) => {
          console.log('handleHypothesesFiltered called with:', hypotheses);
          set((state) => {
            if (!state.currentSession) {
              console.log('No current session, skipping hypotheses filtering');
              return;
            }
            
            console.log('Replacing hypotheses list with filtered results');
            console.log('Before filtering:', state.currentSession.hypotheses.length, 'hypotheses');
            
            // Replace the entire hypotheses array with the filtered list
            state.currentSession.hypotheses = hypotheses.map(h => ({
              ...h,
              isStreaming: false,
              reviews: h.reviews || []
            }));
            
            console.log('After filtering:', state.currentSession.hypotheses.length, 'hypotheses');
          });
          
          get().addNotification({
            type: 'info',
            title: 'Hypotheses Filtered',
            message: `Ranking agent selected top ${hypotheses.length} hypotheses`,
            autoClose: true,
            duration: 3000
          });
        },
        
        // Handle review completion
        handleReviewCompleted: (hypothesisId: string, review: Review) => {
          set((state) => {
            if (!state.currentSession) return;
            
            const hypothesis = state.currentSession.hypotheses.find(
              h => h.id === hypothesisId
            );
            
            if (hypothesis) {
              hypothesis.reviews.push(review);
              state.currentSession.stats.reviewsCompleted++;
            }
          });
        },
        
        // Handle progress updates
        handleProgressUpdate: (stage: string, progress: number, message?: string) => {
          set((state) => {
            if (!state.currentSession) return;
            
            state.currentSession.currentStage = stage;
            state.currentSession.progress = progress;
            
            // Extract current agent from stage
            const agentNames = ['Supervisor', 'Generation', 'Reflection', 'Ranking', 'Evolution', 'Proximity', 'MetaReview'];
            const currentAgent = agentNames.find(agent => 
              stage.toLowerCase().includes(agent.toLowerCase()) || 
              (message && message.toLowerCase().includes(agent.toLowerCase()))
            );
            
            if (currentAgent) {
              state.currentSession.currentAgent = currentAgent;
            }
            
            // Extract iteration from message (look for "Cycle X" pattern)
            if (message) {
              const cycleMatch = message.match(/Cycle (\d+)/);
              if (cycleMatch) {
                state.currentSession.currentIteration = parseInt(cycleMatch[1], 10);
              }
            }
            
            // Set status to active when we receive progress updates
            if (state.currentSession.status === 'initializing') {
              state.currentSession.status = 'active';
            }
            
            if (progress === 100) {
              state.currentSession.status = 'completed';
              state.currentSession.completedAt = new Date().toISOString();
            }
          });
        },
        
        // Handle stage updates
        handleStageUpdate: (stage: string, message: string) => {
          set((state) => {
            if (!state.currentSession) return;
            state.currentSession.currentStage = stage;
            
            // Extract current agent from stage or message
            const agentNames = ['Supervisor', 'Generation', 'Reflection', 'Ranking', 'Evolution', 'Proximity', 'MetaReview'];
            const currentAgent = agentNames.find(agent => 
              stage.toLowerCase().includes(agent.toLowerCase()) || 
              message.toLowerCase().includes(agent.toLowerCase())
            );
            
            if (currentAgent) {
              state.currentSession.currentAgent = currentAgent;
            }
            
            // Extract iteration from message (look for "Cycle X" pattern)
            const cycleMatch = message.match(/Cycle (\d+)/);
            if (cycleMatch) {
              state.currentSession.currentIteration = parseInt(cycleMatch[1], 10);
            }
          });
        },
        
        // Handle errors
        handleError: (error: SystemError) => {
          set((state) => {
            if (state.currentSession) {
              state.currentSession.status = 'error';
              state.currentSession.error = error;
            }
          });
          
          get().addNotification({
            type: 'error',
            title: `Error: ${error.code}`,
            message: error.message,
            autoClose: error.severity === 'low'
          });
        },
        
        // Handle session completion
        handleSessionComplete: (sessionId: string) => {
          set((state) => {
            if (state.currentSession?.id === sessionId) {
              state.currentSession.status = 'completed';
              state.currentSession.completedAt = new Date().toISOString();
              
              // Save to history
              state.sessionHistory.unshift({ ...state.currentSession });
              
              // Keep only last 10 sessions
              if (state.sessionHistory.length > 10) {
                state.sessionHistory = state.sessionHistory.slice(0, 10);
              }
            }
            
            // DON'T reset literature search state when session completes naturally
            // Users should be able to review the papers after research completes
            // Only reset when explicitly starting a new session
          });
          
          get().addNotification({
            type: 'success',
            title: 'Research Complete',
            message: 'AI Co-Scientist has completed the research session',
            autoClose: false
          });
        },
        
        // Handle literature search updates
        handleLiteratureSearchUpdate: (query: string, stage: 'searching' | 'analyzing' | 'complete', papers: Paper[], message: string) => {
          set((state) => {
            state.literatureSearchState.isSearching = stage !== 'complete';
            state.literatureSearchState.query = query;
            state.literatureSearchState.papers = papers;
            state.literatureSearchState.searchStage = stage;
            state.literatureSearchState.message = message;
            
            // Update paper count in session stats
            if (state.currentSession && papers.length > 0) {
              state.currentSession.stats.literaturePapersReviewed = papers.length;
            }
          });
          
          // Add notification for significant updates
          if (stage === 'complete' && papers.length > 0) {
            get().addNotification({
              type: 'info',
              title: 'Literature Search Complete',
              message: `Found ${papers.length} relevant papers`,
              autoClose: true,
              duration: 4000
            });
          }
        },
        
        // Pause research
        pauseResearch: () => {
          get().wsClient?.send({ type: 'pause' });
          set((state) => {
            if (state.currentSession) {
              state.currentSession.status = 'paused';
            }
          });
        },
        
        // Resume research
        resumeResearch: () => {
          get().wsClient?.send({ type: 'resume' });
          set((state) => {
            if (state.currentSession) {
              state.currentSession.status = 'active';
            }
          });
        },
        
        // Stop research
        stopResearch: () => {
          // Show confirmation dialog
          const confirmed = window.confirm(
            'Are you sure you want to stop the research process? This will terminate the current session and cannot be undone.'
          );
          
          if (confirmed) {
            get().stopResearchConfirmed();
          }
        },
        
        // Stop research confirmed (internal function)
        stopResearchConfirmed: () => {
          get().wsClient?.send({ type: 'stop' });
          set((state) => {
            if (state.currentSession) {
              state.currentSession.status = 'completed';
              state.currentSession.completedAt = new Date().toISOString();
            }
            
            // Only reset literature search state if explicitly stopping (not natural completion)
            // Keep the papers visible so users can review them after research completes
            if (state.literatureSearchState.searchStage !== 'complete') {
              state.literatureSearchState = {
                isSearching: false,
                query: '',
                papers: [],
                searchStage: 'idle',
                message: ''
              };
            }
          });
          
          get().addNotification({
            type: 'info',
            title: 'Research Stopped',
            message: 'Research session has been terminated',
            autoClose: true,
            duration: 3000
          });
        },
        
        // Restart research (reset everything)
        restartResearch: () => {
          set((state) => {
            // Reset current session
            state.currentSession = null;
            
            // Reset literature search state
            state.literatureSearchState = {
              isSearching: false,
              query: '',
              papers: [],
              searchStage: 'idle',
              message: ''
            };
            
            // Reset UI state
            state.ui = {
              ...initialUIState,
              mode: state.ui.mode, // Preserve mode
              autoScrollEnabled: state.ui.autoScrollEnabled // Preserve auto-scroll preference
            };
            
            // Clear notifications
            state.notifications = [];
          });
          
          get().addNotification({
            type: 'success',
            title: 'System Reset',
            message: 'Ready for new research session',
            autoClose: true,
            duration: 3000
          });
        },
        
        // Seed new process from hypothesis
        seedNewProcess: (hypothesis: Hypothesis) => {
          // Show confirmation dialog
          const confirmed = window.confirm(
            `Are you sure you want to start a new research process using this hypothesis as the starting point?\n\n"${hypothesis.summary || hypothesis.content.substring(0, 100)}..."\n\nThis will reset the current session.`
          );
          
          if (confirmed) {
            // Reset session state
            get().restartResearch();
            
            // Set the hypothesis content as the new research goal
            const newGoal = `Building on this hypothesis: ${hypothesis.content}`;
            
            // Start new research with the hypothesis as the goal
            const request: StartResearchRequest = {
              goal: newGoal,
              mode: 'simple',
              preferences: {
                maxHypotheses: 5,
                noveltyThreshold: 0.7,
                includeExperimentalProtocols: true,
                prioritizeTestability: true
              }
            };
            
            get().startResearch(request);
            
            get().addNotification({
              type: 'info',
              title: 'New Research Started',
              message: 'Starting new research process based on selected hypothesis',
              autoClose: true,
              duration: 5000
            });
          }
        },
        
        // UI actions
        setMode: (mode: AppMode) => {
          set((state) => {
            state.ui.mode = mode;
          });
        },
        
        setViewMode: (viewMode: ViewMode) => {
          set((state) => {
            state.ui.viewMode = viewMode;
          });
        },
        
        selectHypothesis: (id: string | null) => {
          set((state) => {
            state.ui.selectedHypothesisId = id;
          });
        },
        
        toggleSection: (sectionId: string) => {
          set((state) => {
            if (state.ui.expandedSections.has(sectionId)) {
              state.ui.expandedSections.delete(sectionId);
            } else {
              state.ui.expandedSections.add(sectionId);
            }
          });
        },
        
        setSidebarCollapsed: (collapsed: boolean) => {
          set((state) => {
            state.ui.sidebarCollapsed = collapsed;
          });
        },
        
        toggleDebugInfo: () => {
          set((state) => {
            state.ui.showDebugInfo = !state.ui.showDebugInfo;
          });
        },

        toggleAutoScroll: () => {
          set((state) => {
            state.ui.autoScrollEnabled = !state.ui.autoScrollEnabled;
          });
        },
        
        // Notifications
        addNotification: (notification: Omit<NotificationMessage, 'id' | 'timestamp'>) => {
          const newNotification: NotificationMessage = {
            ...notification,
            id: crypto.randomUUID(),
            timestamp: Date.now()
          };
          
          set((state) => {
            state.notifications.push(newNotification);
            
            // Keep only last 20 notifications
            if (state.notifications.length > 20) {
              state.notifications = state.notifications.slice(-20);
            }
          });
          
          // Auto-remove if specified
          if (notification.autoClose) {
            setTimeout(() => {
              get().removeNotification(newNotification.id);
            }, notification.duration || 5000);
          }
        },
        
        removeNotification: (id: string) => {
          set((state) => {
            const notificationIndex = state.notifications.findIndex(n => n.id === id);
            if (notificationIndex > -1) {
              state.notifications.splice(notificationIndex, 1);
            }
          });
        },
        
        clearNotifications: () => {
          set((state) => {
            state.notifications = [];
          });
        },
        
        // Session management
        saveSession: (session: ResearchSession) => {
          set((state) => {
            const existingIndex = state.sessionHistory.findIndex(s => s.id === session.id);
            if (existingIndex >= 0) {
              state.sessionHistory[existingIndex] = session;
            } else {
              state.sessionHistory.unshift(session);
            }
          });
        },
        
        loadSession: (sessionId: string) => {
          const session = get().sessionHistory.find(s => s.id === sessionId);
          if (session) {
            set((state) => {
              state.currentSession = { ...session };
            });
          }
        },
        
        // Export results
        exportResults: (format: 'json' | 'pdf' | 'csv') => {
          const session = get().currentSession;
          if (!session) return;
          
          // Get literature search state
          const literatureState = get().literatureSearchState;
          
          // Create comprehensive export data including all agent outputs
          const exportData = {
            metadata: {
              session_id: session.id,
              research_goal: session.goal,
              exported_at: new Date().toISOString(),
              export_format: format,
              system_version: "CoScientist v2.0",
              total_iterations: session.currentIteration,
              session_duration_ms: session.stats.executionTimeMs,
              session_status: session.status
            },
            
            research_configuration: {
              preferences: session.preferences,
              mode: get().ui.mode,
              started_at: session.startedAt,
              completed_at: session.completedAt
            },
            
            session_statistics: {
              hypotheses_generated: session.stats.hypothesesGenerated,
              reviews_completed: session.stats.reviewsCompleted,
              literature_papers_reviewed: session.stats.literaturePapersReviewed,
              tournaments_run: session.stats.tournamentsRun || 0,
              execution_time_seconds: Math.round(session.stats.executionTimeMs / 1000),
              agent_activations: session.stats.agentActivations || {}
            },
            
            research_progression: {
              current_iteration: session.currentIteration,
              current_stage: session.currentStage,
              current_agent: session.currentAgent,
              iterations_completed: session.currentIteration >= 6 ? 6 : session.currentIteration,
              research_complete: session.currentIteration >= 6 || session.status === 'completed'
            },
            
            agent_outputs: {
              supervisor: {
                name: "Supervisor",
                description: "Orchestrates the entire research process",
                total_activations: session.stats.agentActivations?.['Supervisor'] || 0,
                output: session.agents.Supervisor?.currentOutput || "",
                status: session.agents.Supervisor?.status || "idle",
                last_message: session.agents.Supervisor?.message || ""
              },
              generation: {
                name: "Generation",
                description: "Creates novel hypotheses using literature review and scientific reasoning",
                total_activations: session.stats.agentActivations?.['Generation'] || 0,
                output: session.agents.Generation?.currentOutput || "",
                status: session.agents.Generation?.status || "idle",
                last_message: session.agents.Generation?.message || "",
                hypotheses_created: session.hypotheses.filter(h => h.createdByAgent === 'Generation').length
              },
              reflection: {
                name: "Reflection",
                description: "Reviews and evaluates hypotheses for quality, novelty, and feasibility",
                total_activations: session.stats.agentActivations?.['Reflection'] || 0,
                output: session.agents.Reflection?.currentOutput || "",
                status: session.agents.Reflection?.status || "idle",
                last_message: session.agents.Reflection?.message || "",
                reviews_completed: session.stats.reviewsCompleted
              },
              ranking: {
                name: "Ranking",
                description: "Runs tournaments to compare and rank hypotheses using multi-dimensional scoring",
                total_activations: session.stats.agentActivations?.['Ranking'] || 0,
                output: session.agents.Ranking?.currentOutput || "",
                status: session.agents.Ranking?.status || "idle",
                last_message: session.agents.Ranking?.message || "",
                tournaments_run: session.stats.tournamentsRun || 0
              },
              evolution: {
                name: "Evolution",
                description: "Improves existing hypotheses through various refinement strategies",
                total_activations: session.stats.agentActivations?.['Evolution'] || 0,
                output: session.agents.Evolution?.currentOutput || "",
                status: session.agents.Evolution?.status || "idle",
                last_message: session.agents.Evolution?.message || "",
                hypotheses_evolved: session.hypotheses.filter(h => h.parentId).length
              },
              proximity: {
                name: "Proximity",
                description: "Clusters similar hypotheses and identifies research patterns",
                total_activations: session.stats.agentActivations?.['Proximity'] || 0,
                output: session.agents.Proximity?.currentOutput || "",
                status: session.agents.Proximity?.status || "idle",
                last_message: session.agents.Proximity?.message || ""
              },
              meta_review: {
                name: "MetaReview",
                description: "Synthesizes system-wide feedback and provides research overview",
                total_activations: session.stats.agentActivations?.['MetaReview'] || 0,
                output: session.agents.MetaReview?.currentOutput || "",
                status: session.agents.MetaReview?.status || "idle",
                last_message: session.agents.MetaReview?.message || ""
              }
            },
            
            literature_search: {
              query_used: literatureState.query,
              search_stage: literatureState.searchStage,
              papers_found: literatureState.papers.length,
              search_message: literatureState.message,
              papers: literatureState.papers.map(paper => ({
                title: paper.title,
                authors: paper.authors,
                year: paper.year,
                citation_count: paper.citationCount || 0,
                abstract: paper.abstract || "",
                url: paper.url || "",
                relevance_score: paper.relevance || 0
              }))
            },
            
            hypotheses: session.hypotheses.map(hypothesis => ({
              id: hypothesis.id,
              rank: hypothesis.rank,
              content: hypothesis.content,
              summary: hypothesis.summary,
              created_by_agent: hypothesis.createdByAgent,
              created_at: hypothesis.createdAt,
              generation: hypothesis.generation,
              parent_id: hypothesis.parentId,
              evolution_method: hypothesis.evolutionMethod,
              scores: {
                novelty: hypothesis.scores.novelty,
                feasibility: hypothesis.scores.feasibility,
                impact: hypothesis.scores.impact,
                testability: hypothesis.scores.testability,
                composite: hypothesis.scores.composite
              },
              supporting_literature: hypothesis.supportingLiterature,
              reviews: hypothesis.reviews.map(review => ({
                id: review.id,
                content: review.content,
                review_type: review.type,
                created_by_agent: review.createdByAgent || 'Unknown',
                created_at: review.createdAt,
                scores: review.scores || {},
                critiques: review.critiques || [],
                suggestions: review.suggestions || []
              })),
              experimental_protocol: hypothesis.experimentalProtocol || {}
            })),
            
            research_insights: {
              top_scoring_hypothesis: session.hypotheses.length > 0 
                ? session.hypotheses.reduce((prev, current) => 
                    (prev.scores.composite > current.scores.composite) ? prev : current
                  )
                : null,
              hypothesis_diversity: {
                total_hypotheses: session.hypotheses.length,
                generations_represented: [...new Set(session.hypotheses.map(h => h.generation))].length,
                agents_contributed: [...new Set(session.hypotheses.map(h => h.createdByAgent))].length,
                evolved_hypotheses: session.hypotheses.filter(h => h.parentId).length
              },
              score_analysis: session.hypotheses.length > 0 ? {
                average_novelty: session.hypotheses.reduce((sum, h) => sum + h.scores.novelty, 0) / session.hypotheses.length,
                average_feasibility: session.hypotheses.reduce((sum, h) => sum + h.scores.feasibility, 0) / session.hypotheses.length,
                average_impact: session.hypotheses.reduce((sum, h) => sum + h.scores.impact, 0) / session.hypotheses.length,
                average_testability: session.hypotheses.reduce((sum, h) => sum + h.scores.testability, 0) / session.hypotheses.length,
                average_composite: session.hypotheses.reduce((sum, h) => sum + h.scores.composite, 0) / session.hypotheses.length
              } : null
            },
            
            system_performance: {
              agents_activated: Object.keys(session.agents).filter(agentName => 
                session.agents[agentName].status !== 'idle' || 
                session.agents[agentName].currentOutput.length > 0
              ).length,
              total_agent_outputs_length: Object.values(session.agents).reduce((total, agent) => 
                total + (agent.currentOutput?.length || 0), 0
              ),
              session_errors: session.error ? [session.error] : [],
              research_completion_percentage: session.currentIteration >= 6 ? 100 : 
                Math.round((session.currentIteration / 6) * 100)
            }
          };
          
          // Generate filename with timestamp
          const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
          const filename = `coscientist-research-${session.id.slice(0, 8)}-${timestamp}.${format}`;
          
          // Download as JSON (for now, regardless of format requested)
          const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
          });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = filename;
          a.click();
          URL.revokeObjectURL(url);
          
          get().addNotification({
            type: 'success',
            title: 'Export Complete',
            message: `Comprehensive research data exported as ${format.toUpperCase()} (${session.hypotheses.length} hypotheses, ${literatureState.papers.length} papers, all agent outputs included)`,
            autoClose: true,
            duration: 5000
          });
        },
        
        // Submit feedback
        submitFeedback: (hypothesisId: string, feedback: any) => {
          // Send via REST API
          fetch('/v1/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              hypothesis_id: hypothesisId,
              ...feedback
            })
          }).then(() => {
            get().addNotification({
              type: 'success',
              title: 'Feedback Submitted',
              message: 'Your feedback has been recorded',
              autoClose: true,
              duration: 3000
            });
          }).catch(() => {
            get().addNotification({
              type: 'error',
              title: 'Feedback Failed',
              message: 'Failed to submit feedback. Please try again.',
              autoClose: true,
              duration: 5000
            });
          });
        }
      })),
      {
        name: 'coscientist-storage',
        partialize: (state) => ({
          sessionHistory: state.sessionHistory,
          ui: {
            mode: state.ui.mode,
            sidebarCollapsed: state.ui.sidebarCollapsed,
            showDebugInfo: state.ui.showDebugInfo
          }
        })
      }
    )
  )
);

// Convenience hooks
export const useCurrentSession = () => 
  useCoScientistStore((state) => state.currentSession);

export const useSelectedHypothesis = () => {
  const selectedId = useCoScientistStore((state) => state.ui.selectedHypothesisId);
  const hypothesis = useCoScientistStore((state) => 
    state.currentSession?.hypotheses.find(h => h.id === selectedId)
  );
  return hypothesis;
};

export const useStreamingAgents = () => 
  useCoScientistStore((state) => state.ui.streamingAgents);

export const useConnectionStatus = () =>
  useCoScientistStore((state) => state.connectionStatus);

export const useNotifications = () =>
  useCoScientistStore((state) => state.notifications);

export const useLiteratureSearchState = () =>
  useCoScientistStore((state) => state.literatureSearchState); 