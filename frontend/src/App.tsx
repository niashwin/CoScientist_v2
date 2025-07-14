import { useEffect, useState } from 'react';
import { useCoScientistStore, useCurrentSession, useStreamingAgents, useConnectionStatus } from './store/useCoScientistStore';
import { MultiAgentStreamingDisplay } from './components/StreamingDisplay';
import { HypothesisCard } from './components/HypothesisCard';
import { AgentPipelineFlowchart } from './components/AgentPipelineFlowchart';
import LiteratureSearchCard from './components/LiteratureSearchCard';
import { StartResearchRequest } from './types';
import { DebugPanel } from './components/DebugPanel';

function App() {
  const [researchGoal, setResearchGoal] = useState('');
  const [isStarting, setIsStarting] = useState(false);
  const [maxIterations, setMaxIterations] = useState(6);
  
  // Store hooks
  const { 
    initializeWebSocket, 
    startResearch, 
    pauseResearch, 
    resumeResearch, 
    stopResearch,
    restartResearch,
    seedNewProcess,
    setMode,
    selectHypothesis,
    exportResults,
    toggleAutoScroll
  } = useCoScientistStore();
  
  const currentSession = useCurrentSession();
  const streamingAgents = useStreamingAgents();
  const connectionStatus = useConnectionStatus();
  const autoScrollEnabled = useCoScientistStore((state) => state.ui.autoScrollEnabled);

  // Initialize WebSocket connection on mount
  useEffect(() => {
    const wsUrl = window.location.hostname === 'localhost' 
      ? 'ws://localhost:8000/ws/auto-run'
      : `wss://${window.location.host}/ws/auto-run`;
    
    initializeWebSocket(wsUrl);
  }, [initializeWebSocket]);

  const handleStartResearch = async () => {
    if (!researchGoal.trim()) return;
    
    setIsStarting(true);
    
    const request: StartResearchRequest = {
      goal: researchGoal.trim(),
      mode: 'simple',
      preferences: {
        maxHypotheses: 5,
        noveltyThreshold: 0.7,
        includeExperimentalProtocols: true,
        prioritizeTestability: true,
        maxIterations: maxIterations
      }
    };
    
    startResearch(request);
    setIsStarting(false);
  };

  const getIterationDescription = (iterations: number) => {
    const descriptions = [
      "Basic hypothesis generation and review",
      "Enhanced with literature search and ranking",
      "Includes hypothesis evolution and refinement",
      "Adds pattern analysis and clustering",
      "Incorporates comprehensive meta-review",
      "Full multi-agent collaboration with tournament ranking",
      "Extended research with deeper analysis",
      "Advanced hypothesis optimization",
      "Comprehensive literature synthesis",
      "Maximum depth research exploration"
    ];
    return descriptions[iterations - 1] || "Extended research exploration";
  };

  const handleModeSwitch = (mode: 'simple' | 'advanced') => {
    setMode(mode);
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-500';
      case 'connecting': return 'text-yellow-500';
      case 'reconnecting': return 'text-orange-500';
      case 'disconnected': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return '🟢 Connected';
      case 'connecting': return '🟡 Connecting...';
      case 'reconnecting': return '🟠 Reconnecting...';
      case 'disconnected': return '🔴 Disconnected';
      default: return '⚪ Unknown';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Title */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">AI</span>
                </div>
                <h1 className="text-xl font-bold text-gray-900">Co-Scientist</h1>
              </div>
              
              {/* Connection Status */}
              <div className={`text-sm ${getConnectionStatusColor()}`}>
                {getConnectionStatusText()}
              </div>
            </div>

            {/* Mode Switch */}
            <div className="flex items-center space-x-4">
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => handleModeSwitch('simple')}
                  className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                    useCoScientistStore.getState().ui.mode === 'simple'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Simple
                </button>
                <button
                  onClick={() => handleModeSwitch('advanced')}
                  className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                    useCoScientistStore.getState().ui.mode === 'advanced'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Advanced
                </button>
              </div>

              {/* Auto-scroll Toggle */}
              <button
                onClick={toggleAutoScroll}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors border ${
                  autoScrollEnabled
                    ? 'bg-blue-50 text-blue-700 border-blue-300'
                    : 'bg-gray-50 text-gray-600 border-gray-300 hover:bg-gray-100'
                }`}
                title={autoScrollEnabled ? 'Disable auto-scroll' : 'Enable auto-scroll'}
              >
                📜 Auto-scroll
              </button>

              {/* Session Controls */}
              {currentSession && (
                <div className="flex items-center space-x-2">
                  {currentSession.status === 'active' && (
                    <button
                      onClick={pauseResearch}
                      className="px-3 py-1 bg-yellow-500 text-white rounded-md text-sm hover:bg-yellow-600 transition-colors"
                    >
                      Pause
                    </button>
                  )}
                  
                  {currentSession.status === 'paused' && (
                    <button
                      onClick={resumeResearch}
                      className="px-3 py-1 bg-green-500 text-white rounded-md text-sm hover:bg-green-600 transition-colors"
                    >
                      Resume
                    </button>
                  )}
                  
                  {/* Stop/Restart Button */}
                  {currentSession.status === 'active' || currentSession.status === 'initializing' ? (
                    <button
                      onClick={stopResearch}
                      className="px-3 py-1 bg-red-500 text-white rounded-md text-sm hover:bg-red-600 transition-colors"
                    >
                      Stop
                    </button>
                  ) : (currentSession.status === 'completed' || currentSession.status === 'error') && (
                    <button
                      onClick={restartResearch}
                      className="px-3 py-1 bg-blue-500 text-white rounded-md text-sm hover:bg-blue-600 transition-colors"
                    >
                      Restart
                    </button>
                  )}
                  
                  <button
                    onClick={() => exportResults('json')}
                    className="px-3 py-1 bg-blue-500 text-white rounded-md text-sm hover:bg-blue-600 transition-colors"
                  >
                    Export
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!currentSession ? (
          /* Welcome Screen */
          <div className="text-center">
            <div className="max-w-3xl mx-auto">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                AI Co-Scientist System
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                Multi-agent system for scientific hypothesis generation
              </p>

              {/* Research Goal Input */}
              <div className="bg-white rounded-lg shadow-md p-6 mb-8">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  What would you like to research?
                </h3>
                
                <div className="space-y-4">
                  <textarea
                    value={researchGoal}
                    onChange={(e) => setResearchGoal(e.target.value)}
                    placeholder="Enter your research goal (e.g., 'Discover novel drug targets for Alzheimer's disease')"
                    className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    rows={3}
                    disabled={connectionStatus !== 'connected'}
                  />
                  
                  {/* Iteration Selector */}
                  <div className="bg-gray-50 rounded-lg p-4 border">
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm font-medium text-gray-700">
                        Research Depth (Iterations)
                      </label>
                      <span className="text-sm text-gray-500">
                        Current: {maxIterations}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-4 mb-3">
                      <input
                        type="range"
                        min="3"
                        max="10"
                        value={maxIterations}
                        onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                        className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        disabled={connectionStatus !== 'connected'}
                      />
                      <div className="flex space-x-1">
                        {[3, 4, 5, 6, 7, 8, 9, 10].map((num) => (
                          <button
                            key={num}
                            onClick={() => setMaxIterations(num)}
                            className={`w-8 h-8 rounded text-xs font-medium transition-colors ${
                              maxIterations === num
                                ? 'bg-blue-500 text-white'
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                            }`}
                            disabled={connectionStatus !== 'connected'}
                          >
                            {num}
                          </button>
                        ))}
                      </div>
                    </div>
                    
                    <div className="text-sm text-gray-600">
                      <div className="font-medium mb-1">What this means:</div>
                      <div className="text-xs leading-relaxed">
                        {getIterationDescription(maxIterations)}
                      </div>
                      <div className="text-xs text-gray-500 mt-2">
                        <strong>Recommended:</strong> 6 iterations for balanced depth and speed. 
                        Higher values provide more thorough analysis but take longer.
                      </div>
                    </div>
                  </div>
                  
                  <button
                    onClick={handleStartResearch}
                    disabled={!researchGoal.trim() || connectionStatus !== 'connected' || isStarting}
                    className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                  >
                    {isStarting ? 'Starting Research...' : 'Start Research'}
                  </button>
                </div>
              </div>

              {/* Feature Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="text-3xl mb-3">🧠</div>
                  <h3 className="font-semibold text-gray-900 mb-2">Multi-Agent Intelligence</h3>
                  <p className="text-gray-600 text-sm">
                    Six specialized AI agents work together to generate, review, and refine hypotheses
                  </p>
                </div>
                
                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="text-3xl mb-3">📚</div>
                  <h3 className="font-semibold text-gray-900 mb-2">Literature Integration</h3>
                  <p className="text-gray-600 text-sm">
                    Automatically searches and incorporates relevant scientific literature
                  </p>
                </div>
                
                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="text-3xl mb-3">⚡</div>
                  <h3 className="font-semibold text-gray-900 mb-2">Real-time Streaming</h3>
                  <p className="text-gray-600 text-sm">
                    Watch AI agents think and generate hypotheses in real-time
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Active Research Session */
          <div className="space-y-8">
            {/* Session Header */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    {currentSession.goal}
                  </h2>
                  <p className="text-gray-600 mt-1">
                    Session started {new Date(currentSession.startedAt).toLocaleString()}
                  </p>
                </div>
                
                <div className="text-right">
                  <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                    currentSession.status === 'active' ? 'bg-green-100 text-green-800' :
                    currentSession.status === 'paused' ? 'bg-yellow-100 text-yellow-800' :
                    currentSession.status === 'completed' ? 'bg-blue-100 text-blue-800' :
                    currentSession.status === 'error' ? 'bg-red-100 text-red-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {currentSession.status}
                  </div>
                </div>
              </div>

              {/* Current Stage Info */}
              <div className="mb-4">
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <span>Current Stage: {currentSession.currentStage}</span>
                </div>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {currentSession.stats.hypothesesGenerated}
                  </div>
                  <div className="text-sm text-gray-600">Hypotheses</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {currentSession.stats.reviewsCompleted}
                  </div>
                  <div className="text-sm text-gray-600">Reviews</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {currentSession.stats.literaturePapersReviewed}
                  </div>
                  <div className="text-sm text-gray-600">Papers</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {Math.round(currentSession.stats.executionTimeMs / 1000)}s
                  </div>
                  <div className="text-sm text-gray-600">Runtime</div>
                </div>
              </div>
            </div>

            {/* Agent Pipeline Flowchart */}
            <AgentPipelineFlowchart
              currentAgent={currentSession.currentAgent}
              currentIteration={currentSession.currentIteration}
            />

            {/* Literature Search Card */}
            <LiteratureSearchCard />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Left Column: Agent Activity */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Agent Activity
                </h3>
                <MultiAgentStreamingDisplay
                  agents={currentSession.agents}
                  streamingAgents={streamingAgents}
                />
              </div>

              {/* Right Column: Generated Hypotheses */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Generated Hypotheses ({currentSession.hypotheses.length})
                </h3>
                
                {currentSession.hypotheses.length === 0 ? (
                  <div className="bg-white rounded-lg shadow-md p-8 text-center text-gray-500">
                    <div className="text-4xl mb-2">🔬</div>
                    <p>No hypotheses generated yet</p>
                    <p className="text-sm mt-1">AI agents are analyzing your research goal...</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {[...currentSession.hypotheses]
                      .sort((a, b) => b.scores.composite - a.scores.composite)
                      .map((hypothesis, index) => (
                        <HypothesisCard
                          key={hypothesis.id}
                          hypothesis={{ ...hypothesis, rank: index + 1 }}
                          isSelected={useCoScientistStore.getState().ui.selectedHypothesisId === hypothesis.id}
                          onSelect={selectHypothesis}
                          showReviews={true}
                          compact={false}
                          showSeedButton={currentSession.status === 'completed'}
                          onSeedNewProcess={seedNewProcess}
                        />
                      ))}
                  </div>
                )}
              </div>
            </div>

            {/* Error Display */}
            {currentSession.error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">
                      Error: {currentSession.error.code}
                    </h3>
                    <p className="mt-1 text-sm text-red-700">
                      {currentSession.error.message}
                    </p>
                    {currentSession.error.suggestions.length > 0 && (
                      <ul className="mt-2 text-sm text-red-700 list-disc list-inside">
                        {currentSession.error.suggestions.map((suggestion, index) => (
                          <li key={index}>{suggestion}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
      
      {/* Debug Panel */}
      <DebugPanel />
    </div>
  );
}

export default App; 