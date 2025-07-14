import React, { useState, useEffect } from 'react';
import { useCoScientistStore } from '../store/useCoScientistStore';

export const DebugPanel: React.FC = () => {
  const [messages, setMessages] = useState<any[]>([]);
  const [isVisible, setIsVisible] = useState(false);
  const currentSession = useCoScientistStore((state) => state.currentSession);
  const connectionStatus = useCoScientistStore((state) => state.connectionStatus);
  const streamingAgents = useCoScientistStore((state) => Array.from(state.ui.streamingAgents));

  useEffect(() => {
    // Listen for WebSocket messages
    const originalConsoleLog = console.log;
    console.log = (...args) => {
      if (args[0]?.includes?.('WebSocket') || args[0]?.includes?.('Received')) {
        setMessages(prev => [...prev.slice(-50), { 
          timestamp: new Date().toISOString(), 
          message: args.join(' '),
          data: args[1] || null
        }]);
      }
      originalConsoleLog.apply(console, args);
    };

    return () => {
      console.log = originalConsoleLog;
    };
  }, []);

  if (!isVisible) {
    return (
      <button
        onClick={() => setIsVisible(true)}
        className="fixed bottom-4 right-4 bg-blue-500 text-white px-4 py-2 rounded shadow-lg z-50"
      >
        Debug Panel
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 w-96 h-96 bg-gray-900 text-white p-4 rounded shadow-lg z-50 overflow-y-auto">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold">Debug Panel</h3>
        <button
          onClick={() => setIsVisible(false)}
          className="text-gray-400 hover:text-white"
        >
          ×
        </button>
      </div>
      
      <div className="mb-4">
        <h4 className="font-semibold">Connection Status:</h4>
        <p className={`text-sm ${connectionStatus === 'connected' ? 'text-green-400' : 'text-red-400'}`}>
          {connectionStatus}
        </p>
      </div>

      <div className="mb-4">
        <h4 className="font-semibold">Session Info:</h4>
        <p className="text-sm">
          Session: {currentSession?.id || 'None'}
        </p>
        <p className="text-sm">
          Hypotheses: {currentSession?.hypotheses?.length || 0}
        </p>
        <p className="text-sm">
          Streaming Agents: {streamingAgents.join(', ') || 'None'}
        </p>
      </div>

      <div className="mb-4">
        <h4 className="font-semibold">Agent States:</h4>
        {currentSession?.agents && Object.entries(currentSession.agents).map(([name, agent]: [string, any]) => (
          <div key={name} className="text-xs mb-2 p-2 bg-gray-800 rounded">
            <div className="font-semibold">{name}</div>
            <div>Status: {agent.status}</div>
            <div>Output Length: {agent.currentOutput?.length || 0}</div>
            <div>Message: {agent.message}</div>
          </div>
        ))}
      </div>

      <div>
        <h4 className="font-semibold">Recent Messages:</h4>
        <div className="text-xs space-y-1 max-h-32 overflow-y-auto">
          {messages.slice(-10).map((msg, index) => (
            <div key={index} className="p-1 bg-gray-800 rounded">
              <div className="text-gray-400">{msg.timestamp}</div>
              <div>{msg.message}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}; 