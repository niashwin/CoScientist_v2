import React from 'react';

interface AgentPipelineFlowchartProps {
  currentAgent?: string;
  currentIteration: number;
  className?: string;
}

const AGENT_PIPELINE = [
  { name: 'Generation', icon: '🧠', description: 'Generate hypotheses' },
  { name: 'Reflection', icon: '🔍', description: 'Review quality' },
  { name: 'Ranking', icon: '🏆', description: 'Tournament ranking' },
  { name: 'Evolution', icon: '🧬', description: 'Refine hypotheses' },
  { name: 'Proximity', icon: '🔗', description: 'Cluster similar ideas' },
  { name: 'MetaReview', icon: '📊', description: 'System-wide feedback' },
];

// Define iteration-aware subtasks (matching backend Literature Query Evolution Agent)
const ITERATION_SUBTASKS = {
  0: "Find recent advances and fundamental mechanisms in the field",
  1: "Identify alternative approaches and methodological variations", 
  2: "Discover gaps, limitations, and unresolved challenges",
  3: "Explore interdisciplinary applications and emerging trends",
  4: "Find experimental protocols and validation methods",
  5: "Identify key researchers and collaborative opportunities"
};

export const AgentPipelineFlowchart: React.FC<AgentPipelineFlowchartProps> = ({ 
  currentAgent, 
  currentIteration, 
  className = '' 
}) => {
  const getCurrentAgentIndex = () => {
    if (!currentAgent) return -1;
    return AGENT_PIPELINE.findIndex(agent => agent.name === currentAgent);
  };

  const currentIndex = getCurrentAgentIndex();
  
  // Get the current subtask description
  const getCurrentSubtask = () => {
    return ITERATION_SUBTASKS[currentIteration as keyof typeof ITERATION_SUBTASKS] || 
           "Find comprehensive literature review and synthesis opportunities";
  };

  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Agent Pipeline
        </h3>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Iteration:</span>
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              {currentIteration}
            </span>
          </div>
        </div>
      </div>

      {/* Current Iteration Subtask */}
      <div className="mb-6 p-4 bg-blue-50 rounded-lg border-l-4 border-blue-400">
        <div className="flex items-start space-x-2">
          <div className="text-blue-600 mt-0.5">🎯</div>
          <div>
            <div className="text-sm font-medium text-blue-900 mb-1">
              Current Research Focus (Iteration {currentIteration}):
            </div>
            <div className="text-sm text-blue-800">
              {getCurrentSubtask()}
            </div>
          </div>
        </div>
      </div>

      <div className="relative">
        {/* Pipeline Flow */}
        <div className="flex items-center justify-between">
          {AGENT_PIPELINE.map((agent, index) => (
            <div key={agent.name} className="flex items-center">
              <div className="flex flex-col items-center space-y-2">
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center text-lg border-2 transition-all duration-300 ${
                    currentAgent === agent.name
                      ? 'bg-blue-500 text-white border-blue-500 shadow-lg scale-110'
                      : currentIndex > index
                      ? 'bg-green-500 text-white border-green-500'
                      : 'bg-gray-200 text-gray-600 border-gray-300'
                  }`}
                >
                  {agent.icon}
                </div>
                <div className="text-center">
                  <div className={`text-sm font-medium ${
                    currentAgent === agent.name ? 'text-blue-600' : 'text-gray-700'
                  }`}>
                    {agent.name}
                  </div>
                  <div className="text-xs text-gray-500 max-w-20">
                    {agent.description}
                  </div>
                </div>
              </div>

              {index < AGENT_PIPELINE.length - 1 && (
                <div className="flex items-center mx-2">
                  <svg
                    className={`w-6 h-6 ${
                      currentIndex > index ? 'text-green-500' : 'text-gray-400'
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Loop back indicator */}
        {currentAgent === 'MetaReview' && currentIteration < 6 && (
          <div className="mt-4 flex items-center justify-center">
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span>Meta-Review will loop back to continue the research cycle (Iteration {currentIteration + 1}/6)</span>
            </div>
          </div>
        )}

        {/* Completion indicator */}
        {currentIteration >= 6 && (
          <div className="mt-4 flex items-center justify-center">
            <div className="flex items-center space-x-2 text-sm text-green-600">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span>Research complete - 6 iterations finished</span>
            </div>
          </div>
        )}

        {/* Progress line */}
        <div className="absolute top-6 left-0 right-0 h-0.5 bg-gray-200 -z-10">
          <div
            className="h-full bg-green-500 transition-all duration-500"
            style={{
              width: currentIndex >= 0 ? `${((currentIndex + 1) / AGENT_PIPELINE.length) * 100}%` : '0%'
            }}
          ></div>
        </div>
      </div>

      {/* Current Agent Status */}
      {currentAgent && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-gray-700">
              Currently active: {currentAgent} Agent
            </span>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {AGENT_PIPELINE.find(agent => agent.name === currentAgent)?.description}
          </div>
        </div>
      )}
    </div>
  );
}; 