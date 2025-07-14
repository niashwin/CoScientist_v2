import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { StreamingDisplayProps } from '../types';
import { useCoScientistStore } from '../store/useCoScientistStore';

export const StreamingDisplay: React.FC<StreamingDisplayProps> = ({
  agent,
  isActive,
  showTimestamp = false,
  maxHeight = '400px',
  defaultExpanded = false  // Changed from true to false
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLPreElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const autoScrollEnabled = useCoScientistStore((state) => state.ui.autoScrollEnabled);
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  console.log('StreamingDisplay rendering agent:', {
    name: agent.name,
    status: agent.status,
    currentOutputLength: agent.currentOutput?.length || 0,
    currentOutputPreview: agent.currentOutput?.substring(0, 100) || '',
    isActive,
    message: agent.message,
    autoScrollEnabled
  });

  // Auto-scroll to bottom within the container when new content arrives
  // Only scroll if the component is currently active/streaming and auto-scroll is enabled
  useEffect(() => {
    if (autoScrollEnabled && agent.status === 'streaming' && contentRef.current && containerRef.current && isExpanded) {
      // Use scrollTo to scroll within the content area only, not affecting page scroll
      const containerElement = containerRef.current;
      
      // Find the scrollable content area
      const scrollableArea = containerElement.querySelector('.streaming-content') as HTMLElement;
      if (scrollableArea) {
        // Smooth scroll to bottom of the content area
        scrollableArea.scrollTo({
          top: scrollableArea.scrollHeight,
          behavior: 'smooth'
        });
      }
    }
  }, [agent.currentOutput, agent.status, autoScrollEnabled, isExpanded]);

  // Update expanded state when defaultExpanded changes
  useEffect(() => {
    setIsExpanded(defaultExpanded);
  }, [defaultExpanded]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'thinking':
        return 'text-yellow-400';
      case 'streaming':
        return 'text-blue-400';
      case 'complete':
        return 'text-green-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'thinking':
        return '🤔';
      case 'streaming':
        return '💭';
      case 'complete':
        return '✅';
      case 'error':
        return '❌';
      default:
        return '⏸️';
    }
  };

  return (
    <div className={`streaming-container bg-gray-900 text-white rounded-lg overflow-hidden transition-all duration-300 ${
      isActive ? 'ring-2 ring-blue-500 shadow-lg shadow-blue-500/20' : 'ring-1 ring-gray-700'
    }`} ref={containerRef}>
      {/* Agent Header - Now Clickable */}
      <div 
        className="agent-header bg-gray-800 px-4 py-3 flex items-center justify-between cursor-pointer hover:bg-gray-750 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-3">
          <span className="text-lg">{getStatusIcon(agent.status)}</span>
          <div>
            <h3 className="font-semibold text-blue-400">{agent.name} Agent</h3>
            <p className={`text-sm ${getStatusColor(agent.status)}`}>
              {agent.status === 'streaming' ? 'Generating...' : 
               agent.status === 'thinking' ? 'Analyzing...' :
               agent.status === 'complete' ? 'Complete' :
               agent.status === 'error' ? 'Error' : 'Idle'}
            </p>
          </div>
        </div>
        
        {/* Status Indicator and Collapse Button */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            {agent.status === 'streaming' && (
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
              </div>
            )}
            
            {showTimestamp && (
              <span className="text-xs text-gray-500">
                {new Date().toLocaleTimeString()}
              </span>
            )}
          </div>
          
          {/* Collapse/Expand Button */}
          <div className="text-gray-400 hover:text-white transition-colors">
            {isExpanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </div>
        </div>
      </div>

      {/* Collapsible Content */}
      {isExpanded && (
        <>
          {/* Status Message */}
          {agent.message && (
            <div className="px-4 py-2 bg-gray-800/50 border-b border-gray-700">
              <p className="text-sm text-gray-300 italic">
                {agent.message}
              </p>
            </div>
          )}

          {/* Content Area */}
          <div 
            className="agent-content streaming-content overflow-y-auto"
            style={{ maxHeight }}
          >
            <pre 
              ref={contentRef}
              className="whitespace-pre-wrap font-mono text-sm p-4 leading-relaxed"
            >
              {agent.currentOutput}
              {agent.status === 'streaming' && (
                <span className="animate-pulse text-blue-400 ml-1">▊</span>
              )}
            </pre>
            <div ref={scrollRef} />
          </div>

          {/* Progress Bar (if available) */}
          {agent.progress !== undefined && (
            <div className="px-4 py-2 bg-gray-800/50">
              <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                <span>Progress</span>
                <span>{Math.round(agent.progress)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1.5">
                <div 
                  className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${agent.progress}%` }}
                ></div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

// Multi-agent streaming display
export const MultiAgentStreamingDisplay: React.FC<{
  agents: Record<string, any>;
  streamingAgents: string[];
  className?: string;
}> = ({ agents, streamingAgents, className = '' }) => {
  const [globalExpanded, setGlobalExpanded] = useState(false); // Changed from true to false
  
  console.log('MultiAgentStreamingDisplay received:', { agents, streamingAgents });
  
  const activeAgents = Object.values(agents).filter(agent => 
    agent.status !== 'idle' || agent.currentOutput.length > 0
  );
  
  console.log('Filtered active agents:', activeAgents);

  if (activeAgents.length === 0) {
    console.log('No active agents, showing placeholder');
    return (
      <div className={`text-center py-8 text-gray-500 ${className}`}>
        <div className="text-4xl mb-2">🧠</div>
        <p>AI agents are ready to assist with your research</p>
      </div>
    );
  }

  console.log('Rendering active agents:', activeAgents.map(a => ({ name: a.name, status: a.status, outputLength: a.currentOutput?.length || 0 })));

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Global Controls */}
      {activeAgents.length > 1 && (
        <div className="flex items-center justify-between mb-4 p-3 bg-gray-100 rounded-lg">
          <div className="text-sm font-medium text-gray-700">
            Agent Outputs ({activeAgents.length})
          </div>
          <button
            onClick={() => setGlobalExpanded(!globalExpanded)}
            className="text-sm text-blue-600 hover:text-blue-800 font-medium flex items-center space-x-1"
          >
            {globalExpanded ? (
              <>
                <ChevronUp className="w-4 h-4" />
                <span>Collapse All</span>
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4" />
                <span>Expand All</span>
              </>
            )}
          </button>
        </div>
      )}
      
      {activeAgents.map((agent) => (
        <StreamingDisplay
          key={agent.name}
          agent={agent}
          isActive={streamingAgents.includes(agent.name)}
          showTimestamp={true}
          defaultExpanded={globalExpanded}
        />
      ))}
    </div>
  );
}; 