from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from swarm import Swarm, Agent
from datetime import datetime
import json
import asyncio
import logging
from pathlib import Path
import openai

from backend.services.llm import LLMService
from backend.services.memory import MemoryService
from backend.services.embeddings import EmbeddingsService
from backend.agents.prompt_optimizer import AdaptivePromptManager
from backend.agents.tournament import EnhancedTournament, EvaluationCriteria
from backend.agents.literature_query_agent import LiteratureQueryEvolutionAgent
from backend.core.config import settings

logger = logging.getLogger(__name__)

class CoScientistSwarm:
    """
    Multi-agent system using OpenAI Swarm for AI co-scientist functionality.
    Orchestrates specialized agents for hypothesis generation, review, and refinement.
    """
    
    def __init__(self, llm_service: LLMService, memory_service: MemoryService, embeddings_service: EmbeddingsService):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.embeddings_service = embeddings_service
        
        # Initialize OpenAI client with API key from settings
        openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.client = Swarm(client=openai_client)
        
        # Initialize optimization and tournament systems
        self.prompt_optimizer = AdaptivePromptManager(memory_service)
        self.tournament_system = EnhancedTournament(llm_service, memory_service)
        
        # Initialize Literature Query Evolution Agent
        self.literature_query_agent = LiteratureQueryEvolutionAgent(llm_service)
        
        # Load prompt templates
        self.prompts = self._load_prompts()
        
        # Initialize agents
        self._initialize_agents()
        
        # Session state
        self.current_session = None
        self.current_session_context = None
        self.stream_callback = None
        
        # Session termination control
        self.session_termination_flags = {}  # session_id -> bool mapping
        
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files"""
        prompts = {}
        prompt_dir = Path(__file__).parent.parent / "prompts"
        
        for prompt_file in prompt_dir.glob("*.prompt"):
            agent_name = prompt_file.stem
            with open(prompt_file, 'r') as f:
                prompts[agent_name] = f.read()
        
        return prompts
    
    def _initialize_agents(self):
        """Initialize all specialized agents using Swarm"""
        
        # Supervisor Agent - orchestrates the entire process
        self.supervisor = Agent(
            name="Supervisor",
            instructions=self.prompts["supervisor"],
            functions=[
                self.route_to_generation,
                self.route_to_reflection,
                self.route_to_ranking,
                self.route_to_evolution,
                self.route_to_proximity,
                self.route_to_meta_review,
                self.check_terminal_state,
                self.update_progress
            ]
        )
        
        # Generation Agent - creates hypotheses
        self.generation_agent = Agent(
            name="Generation",
            instructions=self.prompts["generation"],
            functions=[
                self.search_literature,
                self.generate_hypothesis,
                self.conduct_scientific_debate,
                self.route_to_reflection
            ]
        )
        
        # Reflection Agent - reviews and evaluates hypotheses
        self.reflection_agent = Agent(
            name="Reflection",
            instructions=self.prompts["reflection"],
            functions=[
                self.evaluate_hypothesis,
                self.score_hypothesis,
                self.provide_feedback,
                self.route_to_ranking
            ]
        )
        
        # Ranking Agent - compares and ranks hypotheses
        self.ranking_agent = Agent(
            name="Ranking",
            instructions=self.prompts["ranking"],
            functions=[
                self.run_tournament_match,
                self.compare_hypotheses,
                self.update_rankings,
                self.route_to_evolution
            ]
        )
        
        # Evolution Agent - improves hypotheses
        self.evolution_agent = Agent(
            name="Evolution",
            instructions=self.prompts["evolution"],
            functions=[
                self.improve_feasibility,
                self.enhance_novelty,
                self.refine_mechanism,
                self.improve_testability,
                self.route_to_proximity
            ]
        )
        
        # Proximity Agent - clusters similar hypotheses
        self.proximity_agent = Agent(
            name="Proximity",
            instructions=self.prompts["proximity"],
            functions=[
                self.calculate_similarity,
                self.cluster_hypotheses,
                self.identify_patterns,
                self.route_to_meta_review
            ]
        )
        
        # Meta-Review Agent - synthesizes system-wide feedback
        self.meta_review_agent = Agent(
            name="MetaReview",
            instructions=self.prompts["meta_review"],
            functions=[
                self.synthesize_reviews,
                self.generate_research_overview,
                self.provide_system_feedback,
                self.suggest_improvements
            ]
        )
    
    async def process_research_goal(
        self, 
        goal: str, 
        session_id: str,
        preferences: Dict[str, Any] = None,
        stream_callback: Optional[Callable] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process research goal with streaming output using Swarm orchestration
        
        Args:
            goal: Research goal/question
            session_id: Unique session identifier
            preferences: User preferences for hypothesis generation
            stream_callback: Function to call with streaming updates
            
        Yields:
            Streaming updates from agents
        """
        self.current_session = session_id
        self.stream_callback = stream_callback
        
        # Initialize session context
        context = {
            "goal": goal,
            "session_id": session_id,
            "preferences": preferences or {},
            "system_state": "initializing",
            "task_queue": [],
            "session_context": {
                "hypotheses": [],
                "reviews": [],
                "tournament_results": [],
                "iteration": 0
            }
        }
        
        # Store session context for routing decisions
        self.current_session_context = context["session_context"]
        
        # Track agent outputs for export functionality
        if "agent_outputs" not in context["session_context"]:
            context["session_context"]["agent_outputs"] = {}
        
        # Save session to memory
        from backend.db.models import ResearchGoal
        research_goal = ResearchGoal(
            id=session_id,
            goal_text=goal,
            preferences=preferences or {},
            status="active"
        )
        await self.memory_service.save_research_goal(research_goal)
        
        # Send initial progress update
        yield {
            "type": "progress",
            "stage": "initialization",
            "progress": 10,
            "message": "Initializing AI Co-Scientist system...",
            "sessionId": session_id
        }
        
        # Send stage update
        yield {
            "type": "stage_update",
            "stage": "initialization",
            "message": "Setting up research pipeline..."
        }
        
        if self.stream_callback:
            await self.stream_callback({
                "type": "progress",
                "stage": "initialization",
                "progress": 10,
                "message": "Initializing AI Co-Scientist system...",
                "sessionId": session_id
            })
        
        # Start with supervisor
        messages = [{"role": "user", "content": self._format_supervisor_prompt(context)}]
        current_agent = self.supervisor
        
        try:
            # Send progress for starting generation
            yield {
                "type": "progress",
                "stage": "generation",
                "progress": 25,
                "message": "Starting hypothesis generation...",
                "sessionId": session_id
            }
            
            yield {
                "type": "stage_update",
                "stage": "generation",
                "message": "Generation agent is analyzing the research goal..."
            }
            
            if self.stream_callback:
                await self.stream_callback({
                    "type": "progress",
                    "stage": "generation",
                    "progress": 25,
                    "message": "Starting hypothesis generation...",
                    "sessionId": session_id
                })
            
            iteration = 0
            max_iterations = 50  # Increased significantly to allow multiple cycles
            
            while iteration < max_iterations:
                # Check for session termination
                if self.is_session_terminated(session_id):
                    logger.info(f"Session {session_id} terminated by user request")
                    yield {
                        "type": "session_terminated",
                        "sessionId": session_id,
                        "message": "Research session terminated by user"
                    }
                    break
                
                # Calculate progress based on completed agent cycles and current stage
                # Each complete cycle (Gen->Ref->Rank->Evo->Prox->Meta) is worth progress
                completed_cycles = context["session_context"]["iteration"]
                current_stage_progress = (iteration % 6) * 10  # 10% per stage within a cycle
                base_progress = 25 + (completed_cycles * 30)  # 30% per completed cycle
                progress = min(base_progress + current_stage_progress, 90)  # Cap at 90% until completion
                
                # Send progress update for current agent
                stage_name = current_agent.name.lower()
                cycle_info = f" (Cycle {completed_cycles + 1})" if completed_cycles > 0 else ""
                yield {
                    "type": "progress",
                    "stage": stage_name,
                    "progress": progress,
                    "message": f"{current_agent.name} agent is processing{cycle_info}...",
                    "sessionId": session_id
                }
                
                yield {
                    "type": "stage_update",
                    "stage": stage_name,
                    "message": f"{current_agent.name} agent is analyzing and generating insights{cycle_info}..."
                }
                
                if self.stream_callback:
                    await self.stream_callback({
                        "type": "progress",
                        "stage": stage_name,
                        "progress": progress,
                        "message": f"{current_agent.name} agent is processing{cycle_info}...",
                        "sessionId": session_id
                    })
                
                # Stream agent response and collect the complete response
                complete_response = None
                async for chunk in self._stream_agent_response(current_agent, messages, context):
                    if chunk.get("type") == "agent_response":
                        complete_response = {
                            "agent": chunk["agent"],
                            "response": chunk["response"],
                            "timestamp": chunk["timestamp"]
                        }
                    else:
                        yield chunk
                
                # Ensure we got a complete response
                if not complete_response:
                    raise Exception(f"No complete response received from {current_agent.name}")
                
                # Update context based on response
                context = await self._update_context(context, complete_response)
                self.current_session_context = context["session_context"]
                
                # Check if we should continue or terminate
                if await self._should_terminate(context):
                    logger.info(f"Terminating after {iteration + 1} iterations due to termination conditions")
                    break
                
                # Get next agent from response
                next_agent = self._get_next_agent(complete_response)
                if next_agent:
                    current_agent = next_agent
                    messages = await self._prepare_messages_for_agent(current_agent, context)
                    iteration += 1
                else:
                    logger.info(f"No next agent found after {current_agent.name}, terminating")
                    break
                
                # Safety check to prevent infinite loops
                if iteration >= max_iterations:
                    logger.warning(f"Reached maximum iterations ({max_iterations}), terminating")
                    break
            
            # Send completion progress
            yield {
                "type": "progress",
                "stage": "completion",
                "progress": 100,
                "message": "Research pipeline completed successfully!",
                "sessionId": session_id
            }
            
            yield {
                "type": "stage_update",
                "stage": "completion",
                "message": "Finalizing results and generating summary..."
            }
            
            if self.stream_callback:
                await self.stream_callback({
                    "type": "progress",
                    "stage": "completion",
                    "progress": 100,
                    "message": "Research pipeline completed successfully!",
                    "sessionId": session_id
                })
                    
        except Exception as e:
            logger.error(f"Error in research goal processing: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Current agent: {current_agent.name if current_agent else 'None'}")
            logger.error(f"Context keys: {list(self.current_session_context.keys()) if self.current_session_context else 'None'}")
            
            # Import traceback for better error reporting
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            yield {
                "type": "error",
                "agent": current_agent.name if current_agent else "System",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if self.stream_callback:
                await self.stream_callback({
                    "type": "error",
                    "agent": current_agent.name if current_agent else "System",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Generate final summary
        yield {
            "type": "session_complete",
            "sessionId": session_id,
            "summary": await self._generate_session_summary(context),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.stream_callback:
            await self.stream_callback({
                "type": "session_complete",
                "sessionId": session_id,
                "summary": await self._generate_session_summary(context),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _stream_agent_response(
        self, 
        agent: Agent, 
        messages: List[Dict], 
        context: Dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream agent response with real-time updates and return complete response"""
        
        logger.info(f"Starting streaming response for agent: {agent.name}")
        
        # Notify start of agent processing
        yield {
            "type": "agent_start",
            "agent": agent.name,
            "message": f"{agent.name} is processing...",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Stream the actual response and collect it
        buffer = ""
        chunk_count = 0
        async for chunk in self._get_streaming_response(agent, messages):
            buffer += chunk
            chunk_count += 1
            
            logger.info(f"Streaming chunk {chunk_count} from {agent.name}: {chunk[:50]}...")
            
            # Send chunk to frontend
            stream_message = {
                "type": "stream_chunk",
                "agent": agent.name,
                "content": chunk,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Yielding stream_chunk: {stream_message}")
            yield stream_message
            
            # Don't send via callback since it's already being sent by the generator
            # This was causing duplicate messages
        
        logger.info(f"Completed streaming for {agent.name}, total chunks: {chunk_count}, buffer length: {len(buffer)}")
        
        # Send completion notification
        yield {
            "type": "agent_complete",
            "agent": agent.name,
            "message": f"{agent.name} completed processing",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Return the complete response for processing
        yield {
            "type": "agent_response",
            "agent": agent.name,
            "response": buffer,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _get_streaming_response(self, agent: Agent, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """Get streaming response from agent using LLM service"""
        prompt = self._format_agent_prompt(agent, messages)
        
        logger.info(f"Getting streaming response for {agent.name} with prompt length: {len(prompt)}")
        
        # Use the LLM service's streaming capability
        try:
            logger.info(f"Attempting to use LLM streaming for {agent.name}")
            # Generate streaming response directly
            chunk_count = 0
            async for chunk in self.llm_service.generate_streaming_response(prompt):
                chunk_count += 1
                logger.info(f"LLM streaming chunk {chunk_count} for {agent.name}: {chunk[:30]}...")
                yield chunk
                
            logger.info(f"LLM streaming completed for {agent.name}, total chunks: {chunk_count}")
        except Exception as e:
            # Fallback to non-streaming if streaming fails
            logger.warning(f"Streaming failed for {agent.name}, falling back to non-streaming: {e}")
            full_response = await self.llm_service.generate_response(prompt)
            
            logger.info(f"Fallback response for {agent.name}, length: {len(full_response)}")
            
            # Simulate streaming by yielding chunks
            chunk_size = 10
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i + chunk_size]
                logger.info(f"Simulated streaming chunk for {agent.name}: {chunk}")
                yield chunk
                # Small delay to simulate streaming
                await asyncio.sleep(0.1)
    
    def _format_supervisor_prompt(self, context: Dict) -> str:
        """Format supervisor prompt with current context"""
        return self.prompts["supervisor"].format(
            system_state=context["system_state"],
            research_goal=context["goal"],
            task_queue=json.dumps(context["task_queue"], indent=2),
            session_context=json.dumps(context["session_context"], indent=2)
        )
    
    def _format_agent_prompt(self, agent: Agent, messages: List[Dict]) -> str:
        """Format prompt for specific agent"""
        # Get the last user message as the main prompt
        if messages:
            return messages[-1]["content"]
        return ""
    
    async def _get_agent_response(self, agent: Agent, messages: List[Dict]) -> Dict:
        """Get complete response from agent"""
        prompt = self._format_agent_prompt(agent, messages)
        response = await self.llm_service.generate_response(prompt)
        
        return {
            "agent": agent.name,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _update_context(self, context: Dict, response: Dict) -> Dict:
        """Update session context based on agent response"""
        logger.info(f"Updating context for agent: {response['agent']}")
        
        # Parse response for structured data
        if response["agent"] == "Generation":
            # Extract hypothesis from response
            hypothesis_result = await self._extract_hypothesis(response["response"])
            if hypothesis_result:
                hypotheses = hypothesis_result["hypotheses"]
                logger.info(f"Extracted {len(hypotheses)} hypotheses")
                
                # Add all hypotheses to context
                for hypothesis in hypotheses:
                    context["session_context"]["hypotheses"].append(hypothesis)
                    
                    # Send each hypothesis to frontend
                    if self.stream_callback:
                        await self.stream_callback({
                            "type": "hypothesis_generated",
                            "hypothesis": hypothesis,
                            "sessionId": context["session_id"]
                        })
                
        elif response["agent"] == "Reflection":
            # Extract review from response
            review = await self._extract_review(response["response"])
            if review:
                logger.info(f"Extracted review: {review}")
                context["session_context"]["reviews"].append(review)
                
        elif response["agent"] == "Ranking":
            # Extract tournament results
            tournament_result = await self._extract_tournament_result(response["response"])
            if tournament_result:
                logger.info(f"Extracted tournament result: {tournament_result}")
                context["session_context"]["tournament_results"].append(tournament_result)
                
                # Apply extracted scores to hypotheses
                hypothesis_scores = tournament_result.get("hypothesis_scores", {})
                final_determination = tournament_result.get("final_determination")
                hypotheses = context["session_context"]["hypotheses"]
                
                # Determine which hypotheses were actually compared
                # The ranking agent compares the last two hypotheses in the list
                if len(hypotheses) >= 2:
                    compared_hypotheses = [hypotheses[-2], hypotheses[-1]]
                    logger.info(f"Ranking agent compared hypotheses at positions {len(hypotheses)-1} and {len(hypotheses)} (IDs: {compared_hypotheses[0]['id']}, {compared_hypotheses[1]['id']})")
                    
                    # Map the extracted scores to the actual hypotheses that were compared
                    for ranking_position, scores in hypothesis_scores.items():
                        # ranking_position is 1-based from the ranking agent output
                        # Map it to the actual hypothesis index in our list
                        if ranking_position == 1:
                            actual_hypothesis = compared_hypotheses[0]  # hypotheses[-2]
                        elif ranking_position == 2:
                            actual_hypothesis = compared_hypotheses[1]  # hypotheses[-1]
                        else:
                            logger.warning(f"Unexpected ranking position: {ranking_position}")
                            continue
                        
                        # Update hypothesis scores with extracted values
                        if "novelty" in scores:
                            actual_hypothesis["scores"]["novelty"] = scores["novelty"]
                        if "feasibility" in scores:
                            actual_hypothesis["scores"]["feasibility"] = scores["feasibility"]
                        if "impact" in scores:
                            actual_hypothesis["scores"]["impact"] = scores["impact"]
                        if "testability" in scores:
                            actual_hypothesis["scores"]["testability"] = scores["testability"]
                        if "composite" in scores:
                            actual_hypothesis["scores"]["composite"] = scores["composite"]
                        
                        logger.info(f"Updated scores for hypothesis at ranking position {ranking_position} (ID: {actual_hypothesis['id']}): {actual_hypothesis['scores']}")
                        
                        # Send updated hypothesis to frontend
                        if self.stream_callback:
                            await self.stream_callback({
                                "type": "hypothesis_updated",
                                "hypothesis": actual_hypothesis,
                                "sessionId": context["session_id"]
                            })
                    
                    # Mark the winning hypothesis based on final determination
                    if final_determination:
                        if final_determination == 1:
                            winning_hypothesis = compared_hypotheses[0]  # hypotheses[-2]
                        elif final_determination == 2:
                            winning_hypothesis = compared_hypotheses[1]  # hypotheses[-1]
                        else:
                            logger.warning(f"Unexpected final determination: {final_determination}")
                            winning_hypothesis = None
                        
                        if winning_hypothesis:
                            winning_hypothesis["is_winner"] = True
                            logger.info(f"Marked hypothesis as winner based on final determination {final_determination}: {winning_hypothesis['id']}")
                            
                            # Send updated winning hypothesis to frontend
                            if self.stream_callback:
                                await self.stream_callback({
                                    "type": "hypothesis_updated",
                                    "hypothesis": winning_hypothesis,
                                    "sessionId": context["session_id"]
                                })
                        else:
                            logger.warning(f"Could not find winning hypothesis for final determination {final_determination}")
                else:
                    logger.warning("Not enough hypotheses to apply ranking results")
                
                # Filter hypotheses based on ranking results - keep only top 2
                if len(hypotheses) > 2:
                    # Sort hypotheses by composite score (descending)
                    sorted_hypotheses = sorted(hypotheses, key=lambda h: h["scores"]["composite"], reverse=True)
                    
                    # Keep only the top 2 hypotheses
                    context["session_context"]["hypotheses"] = sorted_hypotheses[:2]
                    
                    logger.info(f"Filtered hypotheses from {len(hypotheses)} to {len(context['session_context']['hypotheses'])} based on ranking")
                    
                    # Log the filtered hypotheses
                    for i, hypothesis in enumerate(context["session_context"]["hypotheses"]):
                        logger.info(f"Kept hypothesis {i+1} (ID: {hypothesis['id']}) with composite score: {hypothesis['scores']['composite']}")
                    
                    # Send filtered hypothesis list to frontend
                    if self.stream_callback:
                        await self.stream_callback({
                            "type": "hypotheses_filtered",
                            "hypotheses": context["session_context"]["hypotheses"],
                            "sessionId": context["session_id"]
                        })
        
        elif response["agent"] == "MetaReview":
            # MetaReview marks the end of a complete cycle
            context["session_context"]["iteration"] += 1
            logger.info(f"Completed research cycle {context['session_context']['iteration']}")
        
        # Update stored session context for routing decisions
        self.current_session_context = context["session_context"]
        
        logger.info(f"Context updated - Hypotheses: {len(context['session_context']['hypotheses'])}, Reviews: {len(context['session_context']['reviews'])}, Iteration: {context['session_context']['iteration']}")
        
        return context
    
    async def _should_terminate(self, context: Dict) -> bool:
        """Determine if session should terminate"""
        # Get max iterations from preferences, default to 6
        preferences = context.get("preferences", {})
        max_iterations = preferences.get("maxIterations", 6)
        
        session_ctx = context["session_context"]
        current_iteration = session_ctx["iteration"]
        
        # Check if we've reached the iteration limit
        if current_iteration >= max_iterations:
            logger.info(f"Terminating - reached iteration limit of {max_iterations} iterations")
            return True
        
        # Get hypothesis and review counts for logging
        hypothesis_count = len(session_ctx["hypotheses"])
        review_count = len(session_ctx["reviews"])
        
        logger.info(f"Continuing - iteration {current_iteration}/{max_iterations}, {hypothesis_count} hypotheses, {review_count} reviews")
        return False
    
    def _get_next_agent(self, response: Dict) -> Optional[Agent]:
        """Determine next agent based on current response and system state"""
        agent_name = response["agent"]
        
        # Get current context to make routing decisions
        session_ctx = self.current_session_context if hasattr(self, 'current_session_context') else {}
        hypothesis_count = len(session_ctx.get("hypotheses", []))
        review_count = len(session_ctx.get("reviews", []))
        iteration = session_ctx.get("iteration", 0)
        
        logger.info(f"Routing from {agent_name}: {hypothesis_count} hypotheses, {review_count} reviews, iteration {iteration}")
        
        # Intelligent routing based on system state
        if agent_name == "Supervisor":
            # Supervisor decides what to do next
            if hypothesis_count < 3:  # Need more hypotheses
                logger.info("Routing to Generation - need more hypotheses")
                return self.generation_agent
            elif review_count < hypothesis_count:
                logger.info("Routing to Reflection - need more reviews")
                return self.reflection_agent  # Review existing hypotheses
            elif len(session_ctx.get("tournament_results", [])) == 0:
                logger.info("Routing to Ranking - need tournament results")
                return self.ranking_agent  # Run tournaments
            else:
                logger.info("Routing to Evolution - ready for evolution")
                return self.evolution_agent  # Evolve best hypotheses
                
        elif agent_name == "Generation":
            # After generation, check if we need to rank new hypotheses
            # If we're in a subsequent iteration (after filtering), we need to rank all hypotheses
            if iteration > 0:
                logger.info("Routing to Ranking after Generation - need to rank new hypotheses in iteration")
                return self.ranking_agent
            else:
                # First iteration - review first, then rank
                logger.info("Routing to Reflection after Generation")
                return self.reflection_agent
            
        elif agent_name == "Reflection":
            # After reflection, check if we need more reviews or can proceed
            if review_count < hypothesis_count:
                logger.info("Routing to Reflection again - more reviews needed")
                return self.reflection_agent  # Continue reviewing
            elif hypothesis_count >= 3:
                logger.info("Routing to Ranking after Reflection")
                return self.ranking_agent  # Rank if we have enough
            else:
                logger.info("Routing to Supervisor after Reflection - need more hypotheses")
                return self.supervisor  # Go back to supervisor for more generation
                
        elif agent_name == "Ranking":
            # After ranking, evolve the best hypotheses
            logger.info("Routing to Evolution after Ranking")
            return self.evolution_agent
            
        elif agent_name == "Evolution":
            # After evolution, analyze patterns
            logger.info("Routing to Proximity after Evolution")
            return self.proximity_agent
            
        elif agent_name == "Proximity":
            # After proximity analysis, ALWAYS do meta-review
            logger.info("Routing to MetaReview after Proximity")
            return self.meta_review_agent
            
        elif agent_name == "MetaReview":
            # Meta-review decides whether to continue or terminate
            # Continue until we reach 6 iterations (hard limit)
            if iteration < 6:  # Continue for more cycles until we hit the 6-iteration limit
                logger.info(f"Routing to Supervisor after MetaReview - continuing (iteration {iteration}/6)")
                return self.supervisor  # Loop back to supervisor for more rounds
            else:
                logger.info(f"Terminating after MetaReview - completed {iteration} iterations (6-iteration limit reached)")
                return None  # Terminate
        
        # Default fallback
        logger.info("Default routing to Supervisor")
        return self.supervisor
    
    async def _prepare_messages_for_agent(self, agent: Agent, context: Dict) -> List[Dict]:
        """Prepare messages for specific agent"""
        if agent.name == "Generation":
            content = await self._format_generation_prompt(context)
            return [{"role": "user", "content": content}]
        elif agent.name == "Reflection":
            return [{"role": "user", "content": self._format_reflection_prompt(context)}]
        elif agent.name == "Ranking":
            return [{"role": "user", "content": self._format_ranking_prompt(context)}]
        elif agent.name == "Evolution":
            return [{"role": "user", "content": self._format_evolution_prompt(context)}]
        elif agent.name == "Proximity":
            return [{"role": "user", "content": self._format_proximity_prompt(context)}]
        elif agent.name == "MetaReview":
            return [{"role": "user", "content": self._format_meta_review_prompt(context)}]
        else:
            return [{"role": "user", "content": self._format_supervisor_prompt(context)}]
    
    async def _refine_literature_query(self, base_query: str, iteration: int, existing_hypotheses: List[Dict]) -> str:
        """Refine literature search query using the Literature Query Evolution Agent"""
        
        try:
            # Use the Literature Query Evolution Agent to generate optimized query
            logger.info(f"Using Literature Query Evolution Agent for iteration {iteration}")
            
            evolved_query = await self.literature_query_agent.generate_evolved_query(
                research_goal=base_query,
                iteration=iteration,
                existing_hypotheses=existing_hypotheses
            )
            
            logger.info(f"Literature Query Agent generated: {evolved_query}")
            return evolved_query
            
        except Exception as e:
            logger.error(f"Error using Literature Query Evolution Agent: {e}")
            
            # Fallback to original simple logic
            logger.info("Falling back to simple query refinement")
            
            # For first iteration, use the base query with academic focus
            if iteration == 0:
                return f"{base_query} recent advances mechanisms filetype:pdf"
            
            # For subsequent iterations, analyze existing hypotheses to refine query
            if existing_hypotheses:
                # Extract key concepts from existing hypotheses
                hypothesis_concepts = []
                for hyp in existing_hypotheses:
                    content = hyp.get('content', '')
                    # Simple concept extraction - look for key scientific terms
                    concepts = self._extract_scientific_concepts(content)
                    hypothesis_concepts.extend(concepts)
                
                # Remove duplicates and get top concepts
                unique_concepts = list(set(hypothesis_concepts))[:3]
                
                if unique_concepts:
                    # Create refined query focusing on gaps and alternatives
                    refined_query = f"{base_query} {' '.join(unique_concepts)} alternative approaches gaps limitations filetype:pdf"
                    return refined_query
            
            # Fallback refinements based on iteration
            refinements = [
                "recent advances mechanisms",
                "alternative approaches novel methods",
                "limitations challenges future directions",
                "interdisciplinary applications emerging trends"
            ]
            
            refinement = refinements[min(iteration, len(refinements) - 1)]
            return f"{base_query} {refinement} filetype:pdf"

    def _extract_scientific_concepts(self, text: str) -> List[str]:
        """Extract scientific concepts from hypothesis text"""
        import re
        
        # Common scientific terms and patterns
        scientific_patterns = [
            r'\b(?:protein|enzyme|receptor|pathway|mechanism|signaling|regulation|expression|activation|inhibition)\b',
            r'\b(?:molecular|cellular|genetic|biochemical|physiological|metabolic|structural|functional)\b',
            r'\b(?:interaction|binding|transport|synthesis|degradation|modification|processing)\b',
            r'\b(?:therapeutic|clinical|diagnostic|prognostic|biomarker|drug|treatment|therapy)\b'
        ]
        
        concepts = []
        text_lower = text.lower()
        
        for pattern in scientific_patterns:
            matches = re.findall(pattern, text_lower)
            concepts.extend(matches)
        
        return concepts[:5]  # Return top 5 concepts

    async def _format_generation_prompt(self, context: Dict) -> str:
        """Format generation agent prompt using AdaptivePromptManager"""
        optimized_prompt = self.prompt_optimizer.get_optimized_prompt("generation", context)
        
        # Import literature logger
        from backend.services.literature_logger import log_websocket_send, log_error
        
        # Perform literature search for this research goal
        literature_papers = []
        session_id = context.get("session_id", "")
        base_query = context["goal"]
        
        # Get current iteration and existing hypotheses
        iteration = context["session_context"].get("iteration", 0)
        existing_hypotheses = context["session_context"].get("hypotheses", [])
        
        # Refine the query based on iteration and existing research
        refined_query = await self._refine_literature_query(base_query, iteration, existing_hypotheses)
        
        try:
            # Send literature search progress update with refined query
            if self.stream_callback:
                await self.stream_callback({
                    "type": "literature_search_update",
                    "stage": "searching",
                    "query": refined_query,
                    "papers": [],
                    "message": f"Searching scientific literature (iteration {iteration + 1})...",
                    "sessionId": session_id
                })
                
                # Log WebSocket send
                log_websocket_send(session_id, refined_query, "searching", [], f"Searching scientific literature (iteration {iteration + 1})...")
            
            # Properly await the literature search with refined query
            literature_papers = await self.search_literature(refined_query)
            
            # Send literature search completion update with papers
            if self.stream_callback:
                await self.stream_callback({
                    "type": "literature_search_update",
                    "stage": "analyzing",
                    "query": refined_query,
                    "papers": literature_papers,
                    "message": f"Found {len(literature_papers)} relevant papers. Analyzing gaps and opportunities...",
                    "sessionId": session_id
                })
                
                # Log WebSocket send
                log_websocket_send(session_id, refined_query, "analyzing", literature_papers, f"Found {len(literature_papers)} relevant papers. Analyzing gaps and opportunities...")
                
                # Send final completion update
                await self.stream_callback({
                    "type": "literature_search_update",
                    "stage": "complete",
                    "query": refined_query,
                    "papers": literature_papers,
                    "message": f"Literature analysis complete. Using {len(literature_papers)} papers for hypothesis generation.",
                    "sessionId": session_id
                })
                
                # Log WebSocket send
                log_websocket_send(session_id, refined_query, "complete", literature_papers, f"Literature analysis complete. Using {len(literature_papers)} papers for hypothesis generation.")
                
        except Exception as e:
            logger.error(f"Literature search failed: {e}")
            
            # Log error
            log_error(session_id, refined_query, str(e), "search_literature")
            
            # Send failure update
            if self.stream_callback:
                await self.stream_callback({
                    "type": "literature_search_update",
                    "stage": "complete",
                    "query": refined_query,
                    "papers": [],
                    "message": "External literature search unavailable. Using internal knowledge base...",
                    "sessionId": session_id
                })
                
                # Log WebSocket send
                log_websocket_send(session_id, refined_query, "complete", [], "External literature search unavailable. Using internal knowledge base...")
        
        # Format literature for prompt
        literature_context = ""
        if literature_papers:
            literature_context = "\n\nRelevant Literature:\n"
            for i, paper in enumerate(literature_papers[:5], 1):  # Use top 5 papers
                literature_context += f"{i}. {paper['title']} ({paper['year']})\n"
                literature_context += f"   Authors: {', '.join(paper['authors'])}\n"
                literature_context += f"   Abstract: {paper['abstract'][:300]}...\n"
                literature_context += f"   Relevance: {paper['relevance']:.2f}\n\n"
        else:
            # Add placeholder literature context to guide hypothesis generation
            literature_context = "\n\nLiterature Context:\n"
            literature_context += "Based on current scientific understanding, focus on identifying novel approaches and gaps in existing research.\n"
            literature_context += "Consider recent advances in related fields and interdisciplinary opportunities.\n\n"
        
        # Determine how many hypotheses to generate
        existing_hypotheses = len(context["session_context"]["hypotheses"])
        target_hypotheses = 3  # Generate 3 hypotheses per iteration (more manageable)
        
        # Create instruction for multiple hypothesis generation
        generation_instruction = f"""
Based on the research goal and available context, generate {target_hypotheses} distinct, novel hypotheses.
Each hypothesis should:
1. Address the research goal from a different angle
2. Be grounded in scientific principles
3. Identify specific gaps or opportunities
4. Propose testable mechanisms or approaches
5. Be clearly distinct from the others

Format your response as:

HYPOTHESIS 1:
[Detailed hypothesis 1]

HYPOTHESIS 2:
[Detailed hypothesis 2]

... and so on for all {target_hypotheses} hypotheses.
"""
        
        return optimized_prompt.format(
            mode="literature_review",
            goal=context["goal"],
            preferences=json.dumps(context.get("preferences", {}), indent=2),
            source_hypothesis="",
            instructions=generation_instruction,
            mode_specific_content=f"Focus on recent literature and identify gaps in current knowledge.{literature_context}"
        )
    
    def _format_reflection_prompt(self, context: Dict) -> str:
        """Format reflection agent prompt using AdaptivePromptManager"""
        hypotheses = context["session_context"]["hypotheses"]
        existing_reviews = context["session_context"]["reviews"]
        
        # Find hypotheses that don't have reviews yet
        reviewed_hypothesis_ids = {review.get("hypothesis_id") for review in existing_reviews}
        unreviewed_hypotheses = [h for h in hypotheses if h.get("id") not in reviewed_hypothesis_ids]
        
        if not unreviewed_hypotheses:
            # If all hypotheses have been reviewed, review the most recent ones again for deeper analysis
            unreviewed_hypotheses = hypotheses[-2:] if len(hypotheses) >= 2 else hypotheses
        
        # Focus on the first unreviewed hypothesis, but provide context about all
        target_hypothesis = unreviewed_hypotheses[0] if unreviewed_hypotheses else {}
        
        optimized_prompt = self.prompt_optimizer.get_optimized_prompt("reflection", context)
        
        # Create context about all hypotheses for better review
        hypotheses_context = ""
        if len(hypotheses) > 1:
            hypotheses_context = "\n\nOther hypotheses for context:\n"
            for i, hyp in enumerate(hypotheses[-3:], 1):  # Show last 3 hypotheses
                if hyp.get("id") != target_hypothesis.get("id"):
                    hypotheses_context += f"{i}. {hyp.get('content', '')[:200]}...\n"
        
        review_instruction = f"""
Please provide a comprehensive review of the following hypothesis:

TARGET HYPOTHESIS:
{target_hypothesis.get('content', '')}

Your review should evaluate:
1. Scientific validity and rigor
2. Novelty and originality
3. Feasibility of testing/implementation
4. Potential impact on the field
5. Clarity and specificity
6. Potential limitations or challenges

Provide scores (1-10) for each dimension and an overall assessment.
{hypotheses_context}

Format your response as:

REVIEW:
[Detailed review text]

SCORES:
Novelty: [1-10]
Feasibility: [1-10]
Impact: [1-10]
Testability: [1-10]
Composite: [calculated average]
"""
        
        return optimized_prompt.format(
            hypothesis=target_hypothesis.get("content", ""),
            literature=json.dumps(target_hypothesis.get("supporting_literature", []), indent=2),
            instructions=review_instruction
        )
    
    def _format_ranking_prompt(self, context: Dict) -> str:
        """Format ranking agent prompt using AdaptivePromptManager"""
        hypotheses = context["session_context"]["hypotheses"]
        
        if len(hypotheses) >= 2:
            # Compare the last two hypotheses (most recently generated)
            hypothesis_1 = hypotheses[-2]
            hypothesis_2 = hypotheses[-1]
            
            comparison_content = f"""
Hypothesis 1 (ID: {hypothesis_1.get('id', 'unknown')}):
{hypothesis_1.get('content', '')}

Hypothesis 2 (ID: {hypothesis_2.get('id', 'unknown')}):
{hypothesis_2.get('content', '')}

Review of hypothesis 1:
{hypothesis_1.get('review', 'No review available')}

Review of hypothesis 2:
{hypothesis_2.get('review', 'No review available')}

Please compare these two hypotheses and provide scores for each on a scale of 1-10 for:
- Scientific Merit
- Novelty and Originality  
- Feasibility and Practicality
- Impact and Significance
- Testability

Format your response with a "Composite Scores:" section containing the individual scores for each hypothesis, followed by your "Final Determination: Better hypothesis: X" where X is 1 or 2.
"""
        else:
            comparison_content = "Insufficient hypotheses for comparison"
        
        optimized_prompt = self.prompt_optimizer.get_optimized_prompt("ranking", context)
        
        return optimized_prompt.format(
            goal=context["goal"],
            preferences=json.dumps(context.get("preferences", {}), indent=2),
            comparison_mode="single_turn",
            notes="Compare based on novelty, feasibility, impact, and testability",
            comparison_content=comparison_content
        )
    
    def _format_evolution_prompt(self, context: Dict) -> str:
        """Format evolution agent prompt"""
        latest_hypothesis = context["session_context"]["hypotheses"][-1] if context["session_context"]["hypotheses"] else {}
        preferences = context.get("preferences", {})
        
        return self.prompts["evolution"].format(
            evolution_strategy="feasibility_improvement",
            goal=context["goal"],
            hypothesis=latest_hypothesis.get("content", ""),
            feedback=latest_hypothesis.get("review", ""),
            guidelines="Improve practical implementability while maintaining novelty",
            preferences=preferences.get("evaluation_criteria", "Novelty, feasibility, and scientific rigor")
        )
    
    def _format_proximity_prompt(self, context: Dict) -> str:
        """Format proximity agent prompt"""
        hypotheses = context["session_context"]["hypotheses"]
        
        return self.prompts["proximity"].format(
            hypotheses=json.dumps([h.get("content", "") for h in hypotheses], indent=2),
            goal=context["goal"],
            criteria="Cluster by mechanism, target system, and experimental approach"
        )
    
    def _format_meta_review_prompt(self, context: Dict) -> str:
        """Format meta-review agent prompt"""
        return self.prompts["meta_review"].format(
            goal=context["goal"],
            preferences=json.dumps(context.get("preferences", {}), indent=2),
            instructions="Synthesize all feedback and provide research overview",
            reviews=json.dumps(context["session_context"]["reviews"], indent=2),
            hypotheses=json.dumps(context["session_context"]["hypotheses"], indent=2),
            tournament_results=json.dumps(context["session_context"]["tournament_results"], indent=2)
        )
    
    async def _extract_hypothesis(self, response: str) -> Optional[Dict]:
        """Extract structured hypothesis from agent response"""
        import uuid
        import re
        
        # Get current iteration for generation tracking
        current_iteration = getattr(self, 'current_session_context', {}).get('iteration', 0)
        
        # Check if response contains multiple hypotheses
        hypothesis_pattern = r'HYPOTHESIS\s+(\d+):\s*(.*?)(?=HYPOTHESIS\s+\d+:|$)'
        matches = re.findall(hypothesis_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Multiple hypotheses found
            hypotheses = []
            for match in matches:
                hypothesis_num, content = match
                content = content.strip()
                
                if content:  # Only create hypothesis if there's content
                    hypothesis = {
                        "id": str(uuid.uuid4()),
                        "content": content,
                        "summary": content[:200] + "..." if len(content) > 200 else content,
                        "scores": {
                            "novelty": 6.5,  # Placeholder - will be replaced by ranking agent
                            "feasibility": 7.2,
                            "impact": 6.8,
                            "testability": 7.0,
                            "composite": 6.9  # Average of above scores
                        },
                        "streamingContent": "",
                        "isStreaming": False,
                        "createdByAgent": "Generation",
                        "supportingLiterature": [],
                        "reviews": [],
                        "generation": current_iteration,
                        "createdAt": datetime.utcnow().isoformat(),
                        "experimentalProtocol": {}
                    }
                    hypotheses.append(hypothesis)
                    logger.info(f"Created hypothesis {hypothesis_num} with ID: {hypothesis['id']} for iteration {current_iteration}")
            
            return {"multiple": True, "hypotheses": hypotheses}
        else:
            # Single hypothesis - original behavior
            hypothesis = {
                "id": str(uuid.uuid4()),
                "content": response,
                "summary": response[:200] + "..." if len(response) > 200 else response,
                "scores": {
                    "novelty": 6.5,  # Placeholder - will be replaced by ranking agent
                    "feasibility": 7.2,
                    "impact": 6.8,
                    "testability": 7.0,
                    "composite": 6.9  # Average of above scores
                },
                "streamingContent": "",
                "isStreaming": False,
                "createdByAgent": "Generation",
                "supportingLiterature": [],
                "reviews": [],
                "generation": current_iteration,
                "createdAt": datetime.utcnow().isoformat(),
                "experimentalProtocol": {}
            }
            
            logger.info(f"Created single hypothesis with ID: {hypothesis['id']} for iteration {current_iteration}")
            return {"multiple": False, "hypotheses": [hypothesis]}
    
    async def _extract_review(self, response: str) -> Optional[Dict]:
        """Extract structured review from agent response"""
        import uuid
        
        # Create a properly formatted review object
        review = {
            "id": str(uuid.uuid4()),
            "content": response,
            "reviewType": "agent_review",
            "critiques": [{"type": "general", "content": response}],
            "suggestions": [],
            "createdByAgent": "Reflection",
            "createdAt": datetime.utcnow().isoformat(),
            "scores": {
                "novelty": 7.5,
                "feasibility": 8.0,
                "impact": 7.0,
                "testability": 8.5
            }
        }
        
        logger.info(f"Created review with ID: {review['id']}")
        return review
    
    async def _extract_tournament_result(self, response: str) -> Optional[Dict]:
        """Extract tournament result from agent response"""
        import re
        
        logger.info("=== STARTING SCORE EXTRACTION ===")
        logger.info(f"Response length: {len(response)} characters")
        
        # Extract composite scores for each hypothesis
        hypothesis_scores = {}
        
        # First, try to find the "Composite Scores:" section which contains individual scores
        composite_section = re.search(r'Composite\s+scores?\s*:\s*(.*?)(?=Final\s+Determination|$)', response, re.DOTALL | re.IGNORECASE)
        if composite_section:
            composite_content = composite_section.group(1)
            logger.info(f"Found composite scores section: {composite_content[:300]}...")
            
            # Look for patterns like "Hypothesis 1 (description):" followed by individual scores
            hypothesis_pattern = r'Hypothesis\s+(\d+)\s*(?:\([^)]+\))?\s*:\s*(.*?)(?=Hypothesis\s+\d+\s*(?:\([^)]+\))?\s*:|$)'
            hypothesis_matches = re.findall(hypothesis_pattern, composite_content, re.DOTALL | re.IGNORECASE)
            
            logger.info(f"Found {len(hypothesis_matches)} hypothesis sections in composite scores")
            
            for match in hypothesis_matches:
                hypothesis_num = int(match[0])
                content = match[1]
                
                logger.info(f"Processing hypothesis {hypothesis_num}, content length: {len(content)}")
                logger.info(f"Content preview: {content[:200]}...")
                
                # Extract individual scores from the content
                scores = {}
                
                # Look for score patterns like "Scientific Merit: 4/5"
                score_patterns = [
                    (r'Scientific\s+Merit:\s*(\d+(?:\.\d+)?)', 'scientific_merit'),
                    (r'Novelty\s+and\s+Originality:\s*(\d+(?:\.\d+)?)', 'novelty'),
                    (r'Feasibility\s+and\s+Practicality:\s*(\d+(?:\.\d+)?)', 'feasibility'),
                    (r'Impact\s+and\s+Significance:\s*(\d+(?:\.\d+)?)', 'impact'),
                    (r'Testability:\s*(\d+(?:\.\d+)?)', 'testability')
                ]
                
                # Extract individual scores
                for pattern, score_name in score_patterns:
                    score_match = re.search(pattern, content, re.IGNORECASE)
                    if score_match:
                        score_value = float(score_match.group(1))
                        # Scores are now expected to be on 1-10 scale directly
                        normalized_score = max(1.0, min(10.0, score_value))
                        scores[score_name] = normalized_score
                        logger.info(f"Extracted {score_name} score for hypothesis {hypothesis_num}: {score_value}/10 -> {normalized_score}/10")
                    else:
                        logger.info(f"No match for {score_name} pattern in hypothesis {hypothesis_num}")
                
                # If we found individual scores, calculate composite and map to frontend names
                if scores:
                    logger.info(f"Found {len(scores)} individual scores for hypothesis {hypothesis_num}")
                    
                    # Map the score names to our frontend names
                    mapped_scores = {
                        'novelty': scores.get('novelty', 0),
                        'feasibility': scores.get('feasibility', 0),
                        'impact': scores.get('impact', 0),
                        'testability': scores.get('testability', 0)
                    }
                    
                    # Calculate composite as average of available scores
                    available_scores = [v for v in mapped_scores.values() if v > 0]
                    if available_scores:
                        composite = sum(available_scores) / len(available_scores)
                        mapped_scores['composite'] = composite
                        
                        # Fill in missing scores with composite value
                        for key in mapped_scores:
                            if mapped_scores[key] == 0:
                                mapped_scores[key] = composite
                        
                        hypothesis_scores[hypothesis_num] = mapped_scores
                        logger.info(f"Calculated scores for hypothesis {hypothesis_num}: {mapped_scores}")
                    else:
                        logger.warning(f"No available scores found for hypothesis {hypothesis_num}")
                else:
                    logger.warning(f"No individual scores found for hypothesis {hypothesis_num}")
        
        # If no composite scores section found, try to find individual scores in the detailed analysis format
        if not hypothesis_scores:
            logger.info("No composite scores section found, trying detailed analysis format...")
            
            # Pattern to match hypothesis evaluation sections
            hypothesis_pattern = r'Hypothesis\s+(\d+)\s*(?:\([^)]+\))?\s*:\s*(.*?)(?=Hypothesis\s+\d+|Final\s+Determination|$)'
            hypothesis_matches = re.findall(hypothesis_pattern, response, re.DOTALL | re.IGNORECASE)
            
            for match in hypothesis_matches:
                hypothesis_num = int(match[0])
                content = match[1]
                
                logger.info(f"Processing hypothesis {hypothesis_num}, content length: {len(content)}")
                logger.info(f"Content preview: {content[:200]}...")
                
                # Extract individual scores from bulleted lists
                scores = {}
                
                # Look for score patterns in bulleted format like "- Scientific Merit: 8"
                bullet_score_patterns = [
                    (r'-\s*Scientific\s+Merit:\s*(\d+(?:\.\d+)?)', 'scientific_merit'),
                    (r'-\s*Novelty\s+and\s+Originality:\s*(\d+(?:\.\d+)?)', 'novelty'),
                    (r'-\s*Feasibility\s+and\s+Practicality:\s*(\d+(?:\.\d+)?)', 'feasibility'),
                    (r'-\s*Impact\s+and\s+Significance:\s*(\d+(?:\.\d+)?)', 'impact'),
                    (r'-\s*Testability:\s*(\d+(?:\.\d+)?)', 'testability')
                ]
                
                # Extract individual scores from bulleted format
                for pattern, score_name in bullet_score_patterns:
                    score_match = re.search(pattern, content, re.IGNORECASE)
                    if score_match:
                        score_value = float(score_match.group(1))
                        # Scores are now expected to be on 1-10 scale directly
                        normalized_score = max(1.0, min(10.0, score_value))
                        scores[score_name] = normalized_score
                        logger.info(f"Extracted {score_name} score for hypothesis {hypothesis_num}: {score_value}/10 -> {normalized_score}/10")
                    else:
                        logger.info(f"No match for {score_name} pattern in hypothesis {hypothesis_num}")
                
                # If we found individual scores, calculate composite and map to frontend names
                if scores:
                    logger.info(f"Found {len(scores)} individual scores for hypothesis {hypothesis_num}")
                    
                    # Map the score names to our frontend names
                    mapped_scores = {
                        'novelty': scores.get('novelty', 0),
                        'feasibility': scores.get('feasibility', 0),
                        'impact': scores.get('impact', 0),
                        'testability': scores.get('testability', 0)
                    }
                    
                    # Calculate composite as average of available scores
                    available_scores = [v for v in mapped_scores.values() if v > 0]
                    if available_scores:
                        composite = sum(available_scores) / len(available_scores)
                        mapped_scores['composite'] = composite
                        
                        # Fill in missing scores with composite value
                        for key in mapped_scores:
                            if mapped_scores[key] == 0:
                                mapped_scores[key] = composite
                        
                        hypothesis_scores[hypothesis_num] = mapped_scores
                        logger.info(f"Calculated scores for hypothesis {hypothesis_num}: {mapped_scores}")
                    else:
                        logger.warning(f"No available scores found for hypothesis {hypothesis_num}")
                else:
                    logger.warning(f"No individual scores found for hypothesis {hypothesis_num}")
        
        # If still no scores found, try the old composite scores format as fallback
        if not hypothesis_scores:
            logger.info("No detailed scores found, trying simple composite scores format...")
            
            # Look for patterns like "Hypothesis 1: 7.8" or "Hypothesis 2: 8.5"
            composite_score_pattern = r'Hypothesis\s+(\d+):\s*(\d+(?:\.\d+)?)'
            composite_matches = re.findall(composite_score_pattern, response, re.IGNORECASE)
            
            for match in composite_matches:
                hypothesis_num = int(match[0])
                raw_score = float(match[1])
                
                # Scores are now expected to be on 1-10 scale directly
                normalized_score = max(1.0, min(10.0, raw_score))
                
                # Create scores dict with composite score and reasonable estimates for individual scores
                scores = {
                    'novelty': normalized_score,
                    'feasibility': normalized_score * 0.9,
                    'impact': normalized_score * 1.1,
                    'testability': normalized_score * 0.95,
                    'composite': normalized_score
                }
                
                # Ensure all scores are within 1-10 range
                for key in scores:
                    scores[key] = max(1.0, min(10.0, scores[key]))
                
                hypothesis_scores[hypothesis_num] = scores
                logger.info(f"Extracted composite score for hypothesis {hypothesis_num}: {raw_score}/10 -> {normalized_score}/10")
        
        # If still no scores, try to extract from individual hypothesis sections with colon format
        if not hypothesis_scores:
            logger.info("No composite scores found, trying individual hypothesis sections...")
            
            # Pattern to match hypothesis evaluation sections
            hypothesis_pattern = r'Hypothesis\s+(\d+)\s*(?:\([^)]+\))?\s*:\s*(.*?)(?=Hypothesis\s+\d+|Final\s+Determination|$)'
            hypothesis_matches = re.findall(hypothesis_pattern, response, re.DOTALL | re.IGNORECASE)
            
            for match in hypothesis_matches:
                hypothesis_num = int(match[0])
                content = match[1]
                
                # Extract individual scores from the content
                scores = {}
                
                # Look for score patterns in the evaluation text
                score_patterns = [
                    (r'Scientific\s+Merit[:\-\s]*(\d+)', 'scientific_merit'),
                    (r'Novelty\s+and\s+Originality[:\-\s]*(\d+)', 'novelty'),
                    (r'Feasibility\s+and\s+Practicality[:\-\s]*(\d+)', 'feasibility'),
                    (r'Impact\s+and\s+Significance[:\-\s]*(\d+)', 'impact'),
                    (r'Testability[:\-\s]*(\d+)', 'testability')
                ]
                
                # Extract individual scores
                for pattern, score_name in score_patterns:
                    score_match = re.search(pattern, content, re.IGNORECASE)
                    if score_match:
                        score_value = float(score_match.group(1))
                        # Scores are now expected to be on 1-10 scale directly
                        normalized_score = max(1.0, min(10.0, score_value))
                        scores[score_name] = normalized_score
                        logger.info(f"Extracted {score_name} score for hypothesis {hypothesis_num}: {score_value}/10 -> {normalized_score}/10")
                
                # Calculate composite score if individual scores are available
                if scores:
                    # Map the score names to our frontend names
                    mapped_scores = {
                        'novelty': scores.get('novelty', 0),
                        'feasibility': scores.get('feasibility', 0),
                        'impact': scores.get('impact', 0),
                        'testability': scores.get('testability', 0)
                    }
                    
                    # Calculate composite as average of available scores
                    available_scores = [v for v in mapped_scores.values() if v > 0]
                    if available_scores:
                        composite = sum(available_scores) / len(available_scores)
                        mapped_scores['composite'] = composite
                        
                        # Fill in missing scores with composite value
                        for key in mapped_scores:
                            if mapped_scores[key] == 0:
                                mapped_scores[key] = composite
                        
                        hypothesis_scores[hypothesis_num] = mapped_scores
                        logger.info(f"Calculated scores for hypothesis {hypothesis_num}: {mapped_scores}")
                    else:
                        logger.warning(f"No available scores found for hypothesis {hypothesis_num}")
                else:
                    logger.warning(f"No individual scores found for hypothesis {hypothesis_num}")
        
        # If still no scores found, try sentiment analysis as final fallback
        if not hypothesis_scores:
            logger.info("No explicit scores found, trying sentiment analysis fallback...")
            
            # Pattern to match hypothesis evaluation sections
            hypothesis_pattern = r'Hypothesis\s+(\d+)\s*(?:\([^)]+\))?\s*:\s*(.*?)(?=Hypothesis\s+\d+|Final\s+Determination|$)'
            hypothesis_matches = re.findall(hypothesis_pattern, response, re.DOTALL | re.IGNORECASE)
            
            for match in hypothesis_matches:
                hypothesis_num = int(match[0])
                content = match[1]
                
                # Extract evaluation sections for sentiment analysis
                evaluation_patterns = [
                    (r'Scientific\s+Merit[:\-\s]*(.*?)(?=Novelty|Feasibility|Impact|Testability|$)', 'scientific_merit'),
                    (r'Novelty\s+and\s+Originality[:\-\s]*(.*?)(?=Feasibility|Impact|Testability|$)', 'novelty'),
                    (r'Feasibility\s+and\s+Practicality[:\-\s]*(.*?)(?=Impact|Testability|$)', 'feasibility'),
                    (r'Impact\s+and\s+Significance[:\-\s]*(.*?)(?=Testability|$)', 'impact'),
                    (r'Testability[:\-\s]*(.*?)(?=Conclusion|$)', 'testability')
                ]
                
                scores = {}
                
                # Try to estimate scores based on positive/negative language
                for pattern, score_name in evaluation_patterns:
                    score_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                    if score_match:
                        evaluation_text = score_match.group(1).lower()
                        
                        # Simple sentiment analysis to estimate score
                        positive_words = ['strength', 'good', 'strong', 'excellent', 'high', 'significant', 'promising', 'feasible', 'testable', 'novel', 'important']
                        negative_words = ['weakness', 'poor', 'weak', 'low', 'limited', 'difficult', 'challenging', 'lack', 'not', 'no', 'absent']
                        
                        positive_count = sum(1 for word in positive_words if word in evaluation_text)
                        negative_count = sum(1 for word in negative_words if word in evaluation_text)
                        
                        # Estimate score based on sentiment (5.5 is neutral, range 1-10)
                        if positive_count > negative_count:
                            estimated_score = 5.5 + (positive_count - negative_count) * 0.8
                        else:
                            estimated_score = 5.5 - (negative_count - positive_count) * 0.8
                        
                        # Clamp to 1-10 range
                        estimated_score = max(1.0, min(10.0, estimated_score))
                        scores[score_name] = estimated_score
                
                # Calculate composite score if individual scores are available
                if scores:
                    # Map the score names to our frontend names
                    mapped_scores = {
                        'novelty': scores.get('novelty', 0),
                        'feasibility': scores.get('feasibility', 0),
                        'impact': scores.get('impact', 0),
                        'testability': scores.get('testability', 0)
                    }
                    
                    # Calculate composite as average of available scores
                    available_scores = [v for v in mapped_scores.values() if v > 0]
                    if available_scores:
                        composite = sum(available_scores) / len(available_scores)
                        mapped_scores['composite'] = composite
                        
                        # Fill in missing scores with composite value
                        for key in mapped_scores:
                            if mapped_scores[key] == 0:
                                mapped_scores[key] = composite
                        
                        hypothesis_scores[hypothesis_num] = mapped_scores
                        logger.info(f"Estimated scores for hypothesis {hypothesis_num}: {mapped_scores}")
                    else:
                        logger.warning(f"No available scores found for hypothesis {hypothesis_num}")
                else:
                    logger.warning(f"No individual scores found for hypothesis {hypothesis_num}")
        
        # Extract the final determination
        final_determination = None
        determination_patterns = [
            r'Final\s+Determination:\s*(?:Better\s+)?Hypothesis:\s*(\d+)',
            r'Final\s+Determination:\s*(?:Better\s+)?hypothesis:\s*(\d+)',
            r'Better\s+hypothesis:\s*(\d+)'
        ]
        
        for pattern in determination_patterns:
            determination_match = re.search(pattern, response, re.IGNORECASE)
            if determination_match:
                final_determination = int(determination_match.group(1))
                logger.info(f"Final determination: Hypothesis {final_determination}")
                break
        
        logger.info(f"=== EXTRACTION COMPLETE ===")
        logger.info(f"Extracted scores for {len(hypothesis_scores)} hypotheses: {hypothesis_scores}")
        logger.info(f"Final determination: {final_determination}")
        
        return {
            "result": response,
            "hypothesis_scores": hypothesis_scores,
            "final_determination": final_determination,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_session_summary(self, context: Dict) -> Dict:
        """Generate final session summary"""
        return {
            "goal": context["goal"],
            "total_hypotheses": len(context["session_context"]["hypotheses"]),
            "total_reviews": len(context["session_context"]["reviews"]),
            "iterations": context["session_context"]["iteration"],
            "top_hypotheses": context["session_context"]["hypotheses"][:3]  # Top 3
        }
    
    # Agent function implementations
    async def route_to_generation(self, context: Dict) -> Agent:
        """Route to generation agent"""
        return self.generation_agent
    
    async def route_to_reflection(self, context: Dict) -> Agent:
        """Route to reflection agent"""
        return self.reflection_agent
    
    async def route_to_ranking(self, context: Dict) -> Agent:
        """Route to ranking agent"""
        return self.ranking_agent
    
    async def route_to_evolution(self, context: Dict) -> Agent:
        """Route to evolution agent"""
        return self.evolution_agent
    
    async def route_to_proximity(self, context: Dict) -> Agent:
        """Route to proximity agent"""
        return self.proximity_agent
    
    async def route_to_meta_review(self, context: Dict) -> Agent:
        """Route to meta-review agent"""
        return self.meta_review_agent
    
    async def check_terminal_state(self, context: Dict) -> bool:
        """Check if session should terminate"""
        return await self._should_terminate(context)
    
    async def update_progress(self, context: Dict, progress: int, stage: str = "processing", message: str = "") -> None:
        """Update session progress"""
        if self.stream_callback:
            await self.stream_callback({
                "type": "progress",
                "stage": stage,
                "progress": progress,
                "message": message,
                "sessionId": context["session_id"]
            })
    
    # Placeholder function implementations for agent tools
    async def search_literature(self, query: str) -> List[Dict]:
        """Search literature for hypothesis generation"""
        from backend.services.literature_search import SmartLiteratureSearch
        from backend.services.fallback_literature import FallbackLiteratureSearch
        from backend.core.config import settings
        from backend.services.literature_logger import log_search_start, log_search_complete, log_fallback_used, log_error
        
        # Get session ID from context (if available)
        session_id = self.current_session
        
        # Log search start
        log_search_start(session_id, query, "search_literature")
        
        # Check if any external API keys are configured
        has_external_apis = any([
            settings.SERPER_API_KEY,
            settings.SEMANTIC_SCHOLAR_API_KEY,
            settings.PERPLEXITY_API_KEY
        ])
        
        try:
            if has_external_apis:
                # Use external literature search services
                literature_service = SmartLiteratureSearch()
                papers = await literature_service.search(query, limit=50)  # Increased from 10 to 50
                
                # Format papers for agent consumption
                formatted_papers = []
                for paper in papers:
                    formatted_papers.append({
                        "title": paper.title,
                        "authors": paper.authors,
                        "year": paper.year,
                        "abstract": paper.abstract,
                        "doi": paper.doi,
                        "relevance": paper.relevance_score,  # Frontend expects 'relevance' not 'relevance_score'
                        "citationCount": paper.citation_count,  # Frontend expects 'citationCount'
                        "url": paper.url,
                        "source": paper.source
                    })
                
                logger.info(f"Found {len(formatted_papers)} papers using external APIs for query: {query}")
                
                # Log search completion
                log_search_complete(session_id, query, formatted_papers, "external_api")
                
                return formatted_papers
            else:
                # Use fallback literature search
                logger.info("No external API keys configured, using fallback literature search")
                fallback_service = FallbackLiteratureSearch()
                papers = await fallback_service.search(query, limit=8)
                
                # Format fallback papers for frontend consistency
                formatted_papers = []
                for paper in papers:
                    formatted_papers.append({
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "year": paper.get("year", 2024),
                        "abstract": paper.get("abstract", ""),
                        "doi": paper.get("doi", ""),
                        "relevance": paper.get("relevance_score", 0.8),
                        "citationCount": paper.get("citation_count", 0),
                        "url": paper.get("url", ""),
                        "source": "fallback"
                    })
                
                logger.info(f"Generated {len(formatted_papers)} fallback literature entries for query: {query}")
                
                # Log fallback used
                log_fallback_used(session_id, query, len(formatted_papers))
                
                # Log search completion
                log_search_complete(session_id, query, formatted_papers, "fallback")
                
                return formatted_papers
                
        except Exception as e:
            logger.error(f"Literature search failed: {e}")
            
            # Log error
            log_error(session_id, query, str(e), "external_api")
            
            # Try fallback as last resort
            try:
                logger.info("Attempting fallback literature search after external API failure")
                fallback_service = FallbackLiteratureSearch()
                papers = await fallback_service.search(query, limit=6)
                
                # Format fallback papers
                formatted_papers = []
                for paper in papers:
                    formatted_papers.append({
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "year": paper.get("year", 2024),
                        "abstract": paper.get("abstract", ""),
                        "doi": paper.get("doi", ""),
                        "relevance": paper.get("relevance_score", 0.8),
                        "citationCount": paper.get("citation_count", 0),
                        "url": paper.get("url", ""),
                        "source": "fallback"
                    })
                
                logger.info(f"Generated {len(papers)} fallback literature entries after API failure")
                
                # Log fallback used
                log_fallback_used(session_id, query, len(formatted_papers))
                
                # Log search completion
                log_search_complete(session_id, query, formatted_papers, "fallback_after_error")
                
                return formatted_papers
            except Exception as fallback_error:
                logger.error(f"Fallback literature search also failed: {fallback_error}")
                
                # Log fallback error
                log_error(session_id, query, str(fallback_error), "fallback")
                
                return []
    
    async def generate_hypothesis(self, context: Dict) -> Dict:
        """Generate new hypothesis"""
        # This is now handled by the main agent loop
        return {}
    
    async def conduct_scientific_debate(self, context: Dict) -> Dict:
        """Conduct scientific debate for hypothesis refinement"""
        # This is now handled by the main agent loop
        return {}
    
    async def evaluate_hypothesis(self, hypothesis: Dict) -> Dict:
        """Evaluate hypothesis quality"""
        # Placeholder implementation
        return {}
    
    async def score_hypothesis(self, hypothesis: Dict) -> Dict:
        """Score hypothesis on multiple dimensions"""
        # Placeholder implementation
        return {}
    
    async def provide_feedback(self, hypothesis: Dict) -> Dict:
        """Provide feedback on hypothesis"""
        # Placeholder implementation
        return {}
    
    async def run_tournament_match(self, hypothesis1: Dict, hypothesis2: Dict) -> Dict:
        """Run tournament match between hypotheses"""
        # Placeholder implementation
        return {}
    
    async def compare_hypotheses(self, hypotheses: List[Dict]) -> Dict:
        """Compare multiple hypotheses"""
        # Placeholder implementation
        return {}
    
    async def update_rankings(self, results: Dict) -> None:
        """Update hypothesis rankings"""
        # Placeholder implementation
        pass
    
    async def improve_feasibility(self, hypothesis: Dict) -> Dict:
        """Improve hypothesis feasibility"""
        # Placeholder implementation
        return {}
    
    async def enhance_novelty(self, hypothesis: Dict) -> Dict:
        """Enhance hypothesis novelty"""
        # Placeholder implementation
        return {}
    
    async def refine_mechanism(self, hypothesis: Dict) -> Dict:
        """Refine hypothesis mechanism"""
        # Placeholder implementation
        return {}
    
    async def improve_testability(self, hypothesis: Dict) -> Dict:
        """Improve hypothesis testability"""
        # Placeholder implementation
        return {}
    
    async def calculate_similarity(self, hypotheses: List[Dict]) -> Dict:
        """Calculate similarity between hypotheses"""
        # Placeholder implementation
        return {}
    
    async def cluster_hypotheses(self, hypotheses: List[Dict]) -> Dict:
        """Cluster similar hypotheses"""
        # Placeholder implementation
        return {}
    
    async def identify_patterns(self, hypotheses: List[Dict]) -> Dict:
        """Identify patterns in hypotheses"""
        # Placeholder implementation
        return {}
    
    async def synthesize_reviews(self, reviews: List[Dict]) -> Dict:
        """Synthesize multiple reviews"""
        # Placeholder implementation
        return {}
    
    async def generate_research_overview(self, context: Dict) -> Dict:
        """Generate research overview"""
        # Placeholder implementation
        return {}
    
    async def provide_system_feedback(self, context: Dict) -> Dict:
        """Provide system-wide feedback"""
        # Placeholder implementation
        return {}
    
    async def suggest_improvements(self, context: Dict) -> Dict:
        """Suggest system improvements"""
        # Placeholder implementation
        return {}
    
    def request_session_termination(self, session_id: str) -> None:
        """Request termination of a specific session"""
        logger.info(f"Termination requested for session {session_id}")
        self.session_termination_flags[session_id] = True
        
    def is_session_terminated(self, session_id: str) -> bool:
        """Check if session termination has been requested"""
        return self.session_termination_flags.get(session_id, False)
        
    def clear_session_termination(self, session_id: str) -> None:
        """Clear termination flag for session"""
        if session_id in self.session_termination_flags:
            del self.session_termination_flags[session_id] 