import React, { useState } from 'react';
import { FileText, Search, CheckCircle, Clock, Users, Calendar, ExternalLink, ChevronDown, ChevronUp } from 'lucide-react';
import { useLiteratureSearchState } from '../store/useCoScientistStore';
import { Paper } from '../types';

interface PaperCardProps {
  paper: Paper;
  index: number;
}

const PaperCard: React.FC<PaperCardProps> = ({ paper, index }) => {
  const truncateText = (text: string, maxLength: number) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center space-x-2">
          <FileText className="w-4 h-4 text-blue-600 flex-shrink-0" />
          <span className="text-sm font-medium text-gray-500">#{index + 1}</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 text-xs text-gray-500">
            <Users className="w-3 h-3" />
            <span>{paper.citationCount || 0}</span>
          </div>
          <div className="flex items-center space-x-1 text-xs text-gray-500">
            <Calendar className="w-3 h-3" />
            <span>{paper.year}</span>
          </div>
        </div>
      </div>
      
      <h4 className="font-medium text-gray-900 mb-2 leading-tight">
        {truncateText(paper.title, 100)}
      </h4>
      
      <div className="text-sm text-gray-600 mb-2">
        <span className="font-medium">Authors:</span> {paper.authors.slice(0, 3).join(', ')}
        {paper.authors.length > 3 && <span className="text-gray-400"> et al.</span>}
      </div>
      
      {paper.abstract && (
        <p className="text-sm text-gray-600 mb-3 leading-relaxed">
          {truncateText(paper.abstract, 200)}
        </p>
      )}
      
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
            Relevance: {(paper.relevance * 100).toFixed(0)}%
          </div>
          {paper.doi && (
            <div className="text-xs text-gray-500">
              DOI: {paper.doi}
            </div>
          )}
        </div>
        
        {paper.url && (
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 text-xs flex items-center space-x-1"
          >
            <ExternalLink className="w-3 h-3" />
            <span>View</span>
          </a>
        )}
      </div>
    </div>
  );
};

const LiteratureSearchCard: React.FC = () => {
  const literatureState = useLiteratureSearchState();
  const [isExpanded, setIsExpanded] = useState(false); // Changed from true to false
  const [showAllPapers, setShowAllPapers] = useState(false);
  
  // Show the card if:
  // 1. Currently searching, OR
  // 2. Search is complete and papers were found, OR
  // 3. Search stage is not idle (meaning a search has been initiated)
  if (!literatureState.isSearching && 
      literatureState.papers.length === 0 && 
      literatureState.searchStage === 'idle') {
    return null;
  }

  const getStatusIcon = () => {
    switch (literatureState.searchStage) {
      case 'searching':
        return <Search className="w-4 h-4 text-blue-600 animate-pulse" />;
      case 'analyzing':
        return <Clock className="w-4 h-4 text-yellow-600 animate-pulse" />;
      case 'complete':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      default:
        return <Search className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusText = () => {
    switch (literatureState.searchStage) {
      case 'searching':
        return 'Searching literature...';
      case 'analyzing':
        return 'Analyzing papers...';
      case 'complete':
        return `Found ${literatureState.papers.length} papers`;
      default:
        return 'Literature search';
    }
  };

  const getStatusColor = () => {
    switch (literatureState.searchStage) {
      case 'searching':
        return 'text-blue-600';
      case 'analyzing':
        return 'text-yellow-600';
      case 'complete':
        return 'text-green-600';
      default:
        return 'text-gray-600';
    }
  };

  const papersToShow = showAllPapers ? literatureState.papers : literatureState.papers.slice(0, 5);

  return (
    <div className="bg-gray-50 rounded-lg border border-gray-200 mb-4">
      {/* Collapsible Header */}
      <div 
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-100 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          {getStatusIcon()}
          <h3 className="text-lg font-semibold text-gray-900">Literature Search</h3>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`text-sm font-medium ${getStatusColor()}`}>
            {getStatusText()}
          </div>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </div>
      
      {/* Collapsible Content */}
      {isExpanded && (
        <div className="px-4 pb-4 border-t border-gray-200">
          {literatureState.query && (
            <div className="mb-4 mt-4">
              <div className="text-sm text-gray-600">
                <span className="font-medium">Query:</span> {literatureState.query}
              </div>
            </div>
          )}
          
          {literatureState.message && (
            <div className="mb-4">
              <div className="text-sm text-gray-700 bg-blue-50 p-3 rounded border-l-4 border-blue-400">
                {literatureState.message}
              </div>
            </div>
          )}
          
          {literatureState.papers.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium text-gray-700">
                  Retrieved Papers ({literatureState.papers.length})
                </div>
                {literatureState.papers.length > 5 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowAllPapers(!showAllPapers);
                    }}
                    className="text-sm text-blue-600 hover:text-blue-800 font-medium"
                  >
                    {showAllPapers ? 'Show Less' : `Show All (${literatureState.papers.length})`}
                  </button>
                )}
              </div>
              
              {/* Scrollable Papers Container */}
              <div className={`space-y-3 ${showAllPapers ? 'max-h-96 overflow-y-auto' : ''}`}>
                {papersToShow.map((paper, index) => (
                  <PaperCard key={`${paper.title}-${index}`} paper={paper} index={index} />
                ))}
              </div>
              
              {!showAllPapers && literatureState.papers.length > 5 && (
                <div className="text-center py-2">
                  <span className="text-sm text-gray-500">
                    Showing 5 of {literatureState.papers.length} papers
                  </span>
                </div>
              )}
            </div>
          )}
          
          {literatureState.isSearching && (
            <div className="mt-4 flex items-center justify-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span className="ml-2 text-sm text-gray-600">Searching for relevant papers...</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LiteratureSearchCard; 