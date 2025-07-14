import React, { useState } from 'react';
import { HypothesisCardProps } from '../types';

export const HypothesisCard: React.FC<HypothesisCardProps> = ({
  hypothesis,
  isSelected,
  onSelect,
  onExpand,
  showReviews = false,
  compact = false,
  showSeedButton = false,
  onSeedNewProcess
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleToggleExpand = () => {
    const newExpanded = !isExpanded;
    setIsExpanded(newExpanded);
    if (onExpand) {
      onExpand(hypothesis.id);
    }
  };

  const getScoreBadgeColor = (score: number) => {
    if (score >= 8) return 'bg-green-500';
    if (score >= 6) return 'bg-yellow-500';
    if (score >= 4) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const formatScore = (score: number) => score.toFixed(1);

  return (
    <div 
      className={`hypothesis-card bg-white rounded-lg shadow-md border transition-all duration-200 cursor-pointer ${
        isSelected 
          ? 'border-blue-500 ring-2 ring-blue-200 shadow-lg' 
          : 'border-gray-200 hover:border-gray-300 hover:shadow-lg'
      } ${compact ? 'p-4' : 'p-6'}`}
      onClick={() => onSelect(hypothesis.id)}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-2">
            <h3 className={`font-semibold text-gray-900 ${compact ? 'text-lg' : 'text-xl'}`}>
              Hypothesis #{hypothesis.rank || 'New'}
            </h3>
            
            {/* Composite Score Badge */}
            <div className={`px-3 py-1 rounded-full text-white text-sm font-medium ${
              getScoreBadgeColor(hypothesis.scores.composite)
            }`}>
              {formatScore(hypothesis.scores.composite)}/10
            </div>
            
            {/* Streaming Indicator */}
            {hypothesis.isStreaming && (
              <div className="flex items-center space-x-2 text-blue-500">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <span className="text-sm">Generating...</span>
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <span>by {hypothesis.createdByAgent}</span>
            <span>•</span>
            <span>{new Date(hypothesis.createdAt).toLocaleString()}</span>
            {hypothesis.generation > 0 && (
              <>
                <span>•</span>
                <span>Gen {hypothesis.generation}</span>
              </>
            )}
          </div>
        </div>
        
        {/* Expand Button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            handleToggleExpand();
          }}
          className="text-gray-400 hover:text-gray-600 transition-colors"
        >
          <svg 
            className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Summary */}
      <div className="mb-4">
        <p className={`text-gray-700 ${compact ? 'text-sm' : 'text-base'} leading-relaxed`}>
          {hypothesis.summary || hypothesis.content.substring(0, 200) + '...'}
        </p>
      </div>

      {/* Scores Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <ScoreItem 
          label="Novelty" 
          score={hypothesis.scores.novelty} 
          compact={compact}
        />
        <ScoreItem 
          label="Feasibility" 
          score={hypothesis.scores.feasibility} 
          compact={compact}
        />
        <ScoreItem 
          label="Impact" 
          score={hypothesis.scores.impact} 
          compact={compact}
        />
        <ScoreItem 
          label="Testability" 
          score={hypothesis.scores.testability} 
          compact={compact}
        />
      </div>

      {/* Tags/Metadata */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {hypothesis.supportingLiterature.length > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              📚 {hypothesis.supportingLiterature.length} papers
            </span>
          )}
          
          {hypothesis.reviews.length > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
              📝 {hypothesis.reviews.length} reviews
            </span>
          )}
          
          {hypothesis.experimentalProtocol && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              🧪 Protocol
            </span>
          )}
        </div>
        
        {/* Action Buttons */}
        <div className="flex items-center space-x-2">
          <button className="text-gray-400 hover:text-blue-500 transition-colors">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
          </button>
          <button className="text-gray-400 hover:text-green-500 transition-colors">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
            </svg>
          </button>
          
          {/* Seed New Process Button */}
          {showSeedButton && onSeedNewProcess && (
            <button 
              onClick={(e) => {
                e.stopPropagation();
                onSeedNewProcess(hypothesis);
              }}
              className="text-gray-400 hover:text-purple-500 transition-colors"
              title="Use this hypothesis to start a new research process"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="mt-6 pt-6 border-t border-gray-200">
          {/* Full Content */}
          <div className="mb-6">
            <h4 className="font-medium text-gray-900 mb-3">Full Hypothesis</h4>
            <div className="prose prose-sm max-w-none">
              <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                {hypothesis.streamingContent || hypothesis.content}
              </p>
            </div>
          </div>

          {/* Detailed Score Analysis */}
          <div className="mb-6">
            <h4 className="font-medium text-gray-900 mb-3">Detailed Score Analysis</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { label: 'Novelty', score: hypothesis.scores.novelty, icon: '💡' },
                { label: 'Feasibility', score: hypothesis.scores.feasibility, icon: '🔬' },
                { label: 'Impact', score: hypothesis.scores.impact, icon: '🚀' },
                { label: 'Testability', score: hypothesis.scores.testability, icon: '🧪' }
              ].map(({ label, score, icon }) => {
                const getDetailedDescription = (label: string, score: number) => {
                  switch (label) {
                    case 'Novelty':
                      return {
                        description: score >= 8 ? 'This hypothesis presents groundbreaking new concepts that could revolutionize current understanding.' :
                                   score >= 6 ? 'The hypothesis offers novel insights or methodological approaches with clear originality.' :
                                   score >= 4 ? 'Contains some original elements but builds on established knowledge.' :
                                   'Limited novelty - primarily restates or slightly modifies existing concepts.',
                        factors: ['Originality of core concepts', 'Departure from existing theories', 'Potential for paradigm shift', 'Uniqueness of proposed mechanisms']
                      };
                    case 'Feasibility':
                      return {
                        description: score >= 8 ? 'Highly feasible with current technology and readily available resources.' :
                                   score >= 6 ? 'Feasible with standard laboratory equipment and reasonable funding.' :
                                   score >= 4 ? 'Challenging but achievable with specialized resources or extended timeline.' :
                                   'Difficult to test with current technology or requires prohibitive resources.',
                        factors: ['Available technology', 'Resource requirements', 'Technical complexity', 'Time constraints']
                      };
                    case 'Impact':
                      return {
                        description: score >= 8 ? 'Could fundamentally transform the field and have broad scientific implications.' :
                                   score >= 6 ? 'Significant potential to advance understanding and influence future research.' :
                                   score >= 4 ? 'Moderate contribution to the field with some practical applications.' :
                                   'Limited impact - unlikely to substantially change current understanding.',
                        factors: ['Scientific significance', 'Practical applications', 'Influence on future research', 'Broader implications']
                      };
                    case 'Testability':
                      return {
                        description: score >= 8 ? 'Makes clear, specific, and measurable predictions that can be definitively tested.' :
                                   score >= 6 ? 'Well-defined predictions with clear experimental approaches.' :
                                   score >= 4 ? 'Testable but may require creative experimental design or interpretation.' :
                                   'Vague or difficult to test - lacks clear falsifiable predictions.',
                        factors: ['Specificity of predictions', 'Measurability of outcomes', 'Falsifiability', 'Experimental clarity']
                      };
                    default:
                      return { description: '', factors: [] };
                  }
                };

                const details = getDetailedDescription(label, score);
                const getScoreColor = (score: number) => {
                  if (score >= 8) return 'text-green-600 bg-green-50 border-green-200';
                  if (score >= 6) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
                  if (score >= 4) return 'text-orange-600 bg-orange-50 border-orange-200';
                  return 'text-red-600 bg-red-50 border-red-200';
                };

                return (
                  <div key={label} className={`p-4 rounded-lg border ${getScoreColor(score)}`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-lg">{icon}</span>
                        <span className="font-medium">{label}</span>
                      </div>
                      <span className="text-lg font-bold">{score.toFixed(1)}/10</span>
                    </div>
                    <p className="text-sm mb-3 leading-relaxed">{details.description}</p>
                    <div className="text-xs opacity-75">
                      <div className="font-medium mb-1">Key factors:</div>
                      <ul className="list-disc list-inside space-y-0.5">
                        {details.factors.map((factor, index) => (
                          <li key={index}>{factor}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Supporting Literature */}
          {hypothesis.supportingLiterature.length > 0 && (
            <div className="mb-6">
              <h4 className="font-medium text-gray-900 mb-3">Supporting Literature</h4>
              <div className="space-y-2">
                {hypothesis.supportingLiterature.slice(0, 3).map((paper, index) => (
                  <div key={index} className="p-3 bg-gray-50 rounded-lg">
                    <h5 className="font-medium text-sm text-gray-900">{paper.title}</h5>
                    <p className="text-xs text-gray-600 mt-1">
                      {paper.authors.join(', ')} ({paper.year})
                    </p>
                    <div className="flex items-center mt-2">
                      <span className="text-xs text-blue-600">
                        Relevance: {(paper.relevance * 100).toFixed(0)}%
                      </span>
                      {paper.citationCount && (
                        <span className="text-xs text-gray-500 ml-3">
                          {paper.citationCount} citations
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Reviews */}
          {showReviews && hypothesis.reviews.length > 0 && (
            <div className="mb-6">
              <h4 className="font-medium text-gray-900 mb-3">Reviews</h4>
              <div className="space-y-3">
                {hypothesis.reviews.map((review) => (
                  <div key={review.id} className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-900">
                        {review.createdByAgent} • {review.type}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(review.createdAt).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">
                      {review.content.substring(0, 200)}...
                    </p>
                    {review.noveltyAssessment && (
                      <div className="mt-2">
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                          review.noveltyAssessment === 'breakthrough' ? 'bg-green-100 text-green-800' :
                          review.noveltyAssessment === 'novel' ? 'bg-blue-100 text-blue-800' :
                          review.noveltyAssessment === 'incremental' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {review.noveltyAssessment}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Experimental Protocol */}
          {hypothesis.experimentalProtocol && (
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Experimental Protocol</h4>
              <div className="p-4 bg-green-50 rounded-lg">
                <h5 className="font-medium text-green-900 mb-2">
                  {hypothesis.experimentalProtocol.title}
                </h5>
                <p className="text-sm text-green-800 mb-3">
                  {hypothesis.experimentalProtocol.objective}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <h6 className="font-medium text-green-900">Timeline</h6>
                    <p className="text-green-800">{hypothesis.experimentalProtocol.timeline}</p>
                  </div>
                  {hypothesis.experimentalProtocol.budget && (
                    <div>
                      <h6 className="font-medium text-green-900">Budget</h6>
                      <p className="text-green-800">{hypothesis.experimentalProtocol.budget}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Score Item Component
const ScoreItem: React.FC<{
  label: string;
  score: number;
  compact?: boolean;
}> = ({ label, score, compact = false }) => {
  const getScoreColor = (score: number) => {
    if (score >= 8) return 'text-green-600 bg-green-50 border-green-200';
    if (score >= 6) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    if (score >= 4) return 'text-orange-600 bg-orange-50 border-orange-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getScoreDescription = (label: string, score: number) => {
    const getQualitativeScore = (score: number) => {
      if (score >= 8.5) return 'Excellent';
      if (score >= 7) return 'Good';
      if (score >= 5.5) return 'Moderate';
      if (score >= 4) return 'Fair';
      return 'Poor';
    };

    const qualitative = getQualitativeScore(score);
    
    switch (label) {
      case 'Novelty':
        return {
          description: `${qualitative} originality - ${score >= 8 ? 'Groundbreaking new approach' : 
                       score >= 6 ? 'Novel insights or methods' : 
                       score >= 4 ? 'Some original elements' : 'Limited novelty'}`,
          details: 'Measures how original and innovative the hypothesis is compared to existing knowledge'
        };
      case 'Feasibility':
        return {
          description: `${qualitative} feasibility - ${score >= 8 ? 'Easily testable with current methods' : 
                       score >= 6 ? 'Testable with available resources' : 
                       score >= 4 ? 'Challenging but possible' : 'Difficult to test'}`,
          details: 'Evaluates how practical it is to test this hypothesis with current technology and resources'
        };
      case 'Impact':
        return {
          description: `${qualitative} impact - ${score >= 8 ? 'Could transform the field' : 
                       score >= 6 ? 'Significant advancement potential' : 
                       score >= 4 ? 'Moderate contribution' : 'Limited impact'}`,
          details: 'Assesses the potential significance and influence on scientific understanding or applications'
        };
      case 'Testability':
        return {
          description: `${qualitative} testability - ${score >= 8 ? 'Clear, measurable predictions' : 
                       score >= 6 ? 'Well-defined test criteria' : 
                       score >= 4 ? 'Testable with some effort' : 'Vague or hard to test'}`,
          details: 'Measures how clearly the hypothesis can be tested and potentially falsified'
        };
      default:
        return {
          description: `${qualitative} score`,
          details: 'Overall assessment score'
        };
    }
  };

  const scoreInfo = getScoreDescription(label, score);

  return (
    <div 
      className={`score-item p-3 rounded-lg border transition-all hover:shadow-md cursor-help ${getScoreColor(score)}`}
      title={scoreInfo.details}
    >
      <div className="flex items-center justify-between mb-1">
        <div className={`font-bold ${compact ? 'text-sm' : 'text-lg'}`}>
          {score.toFixed(1)}
        </div>
        <div className={`text-xs opacity-60 ${compact ? 'text-xs' : 'text-sm'}`}>
          /10
        </div>
      </div>
      <div className={`font-medium ${compact ? 'text-xs' : 'text-sm'} mb-1`}>
        {label}
      </div>
      {!compact && (
        <div className="text-xs opacity-75 leading-tight">
          {scoreInfo.description}
        </div>
      )}
    </div>
  );
}; 