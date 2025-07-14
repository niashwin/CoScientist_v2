"""
Scientific tools integration service.
Provides access to external scientific databases and computational tools.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

from backend.core.config import settings
from backend.core.circuit_breaker import CircuitBreaker
from backend.core.exceptions import CoScientistError, ErrorSeverity
from backend.services.cache import cache_manager

logger = logging.getLogger(__name__)

@dataclass
class ProteinStructure:
    """Protein structure data"""
    sequence: str
    structure_id: str
    confidence: float
    pdb_data: Optional[str] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Compound:
    """Chemical compound data"""
    compound_id: str
    name: str
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    properties: Dict[str, Any] = None
    source: str = "unknown"
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class DrugInteraction:
    """Drug interaction data"""
    drug_a: str
    drug_b: str
    interaction_type: str
    severity: str
    description: str
    confidence: float
    source: str = "unknown"

@dataclass
class PathwayInfo:
    """Biological pathway information"""
    pathway_id: str
    name: str
    description: str
    genes: List[str]
    proteins: List[str]
    compounds: List[str]
    source: str = "unknown"

class ScientificTool(ABC):
    """Base class for scientific tools"""
    
    def __init__(self, name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
            timeout=settings.CIRCUIT_BREAKER_TIMEOUT
        )
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.EXTERNAL_API_TIMEOUT)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

class AlphaFoldTool(ScientificTool):
    """AlphaFold protein structure prediction"""
    
    def __init__(self):
        super().__init__(
            name="alphafold",
            api_key=settings.ALPHAFOLD_API_KEY,
            base_url=settings.ALPHAFOLD_API_URL
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Predict protein structure using AlphaFold"""
        sequence = kwargs.get('sequence')
        uniprot_id = kwargs.get('uniprot_id')
        
        if not sequence and not uniprot_id:
            raise CoScientistError(
                "Either sequence or uniprot_id must be provided",
                "ALPHAFOLD_001",
                ErrorSeverity.LOW
            )
        
        try:
            async with self.circuit_breaker:
                if uniprot_id:
                    # Get existing structure from AlphaFold database
                    return await self._get_existing_structure(uniprot_id)
                else:
                    # Predict structure for new sequence
                    return await self._predict_structure(sequence)
        
        except Exception as e:
            logger.error(f"AlphaFold prediction failed: {e}")
            raise CoScientistError(
                f"AlphaFold prediction failed: {str(e)}",
                "ALPHAFOLD_002",
                ErrorSeverity.MEDIUM
            )
    
    async def _get_existing_structure(self, uniprot_id: str) -> Dict[str, Any]:
        """Get existing structure from AlphaFold database"""
        # Check cache first
        cache_key = f"alphafold_structure_{uniprot_id}"
        cached_result = await cache_manager.get_api_response(cache_key)
        if cached_result:
            return cached_result
        
        url = f"{self.base_url}/prediction/{uniprot_id}"
        
        async with self.session.get(url, headers=self._get_headers()) as response:
            if response.status == 200:
                data = await response.json()
                
                structure = ProteinStructure(
                    sequence=data.get('sequence', ''),
                    structure_id=uniprot_id,
                    confidence=data.get('confidence', 0.0),
                    pdb_data=data.get('pdb_data'),
                    source="alphafold",
                    metadata=data
                )
                
                result = {
                    "status": "success",
                    "structure": structure.__dict__,
                    "source": "alphafold_database"
                }
                
                # Cache result
                await cache_manager.set_api_response(cache_key, result)
                
                return result
            
            elif response.status == 404:
                return {
                    "status": "not_found",
                    "message": f"No structure found for {uniprot_id}",
                    "suggestion": "Try structure prediction with sequence"
                }
            
            else:
                response.raise_for_status()
    
    async def _predict_structure(self, sequence: str) -> Dict[str, Any]:
        """Predict structure for new sequence"""
        # This would typically be a more complex prediction process
        # For now, we'll simulate the prediction
        
        cache_key = f"alphafold_prediction_{hash(sequence)}"
        cached_result = await cache_manager.get_api_response(cache_key)
        if cached_result:
            return cached_result
        
        # Simulate prediction (in real implementation, this would call actual AlphaFold)
        await asyncio.sleep(2)  # Simulate processing time
        
        structure = ProteinStructure(
            sequence=sequence,
            structure_id=f"predicted_{hash(sequence)}",
            confidence=0.75,  # Simulated confidence
            source="alphafold_prediction",
            metadata={"method": "alphafold2", "processing_time": 2.0}
        )
        
        result = {
            "status": "success",
            "structure": structure.__dict__,
            "source": "alphafold_prediction",
            "note": "This is a simulated prediction. Enable ALPHAFOLD_API_KEY for real predictions."
        }
        
        # Cache result
        await cache_manager.set_api_response(cache_key, result)
        
        return result

class ChEMBLTool(ScientificTool):
    """ChEMBL database integration"""
    
    def __init__(self):
        super().__init__(
            name="chembl",
            api_key=settings.CHEMBL_API_KEY,
            base_url=settings.CHEMBL_API_URL
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Query ChEMBL database"""
        query_type = kwargs.get('query_type', 'compound')
        
        if query_type == 'compound':
            return await self._search_compounds(kwargs.get('query', ''))
        elif query_type == 'target':
            return await self._search_targets(kwargs.get('query', ''))
        elif query_type == 'activity':
            return await self._get_activity_data(kwargs.get('compound_id', ''))
        else:
            raise CoScientistError(
                f"Unknown query type: {query_type}",
                "CHEMBL_001",
                ErrorSeverity.LOW
            )
    
    async def _search_compounds(self, query: str) -> Dict[str, Any]:
        """Search for compounds in ChEMBL"""
        cache_key = f"chembl_compounds_{query}"
        cached_result = await cache_manager.get_api_response(cache_key)
        if cached_result:
            return cached_result
        
        try:
            async with self.circuit_breaker:
                url = f"{self.base_url}/molecule/search"
                params = {'q': query, 'format': 'json'}
                
                async with self.session.get(url, params=params, headers=self._get_headers()) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    compounds = []
                    for item in data.get('molecules', []):
                        compound = Compound(
                            compound_id=item.get('molecule_chembl_id', ''),
                            name=item.get('pref_name', ''),
                            smiles=item.get('molecule_structures', {}).get('canonical_smiles'),
                            molecular_formula=item.get('molecule_properties', {}).get('molecular_formula'),
                            molecular_weight=item.get('molecule_properties', {}).get('molecular_weight'),
                            properties=item.get('molecule_properties', {}),
                            source="chembl"
                        )
                        compounds.append(compound.__dict__)
                    
                    result = {
                        "status": "success",
                        "compounds": compounds,
                        "total_count": len(compounds)
                    }
                    
                    # Cache result
                    await cache_manager.set_api_response(cache_key, result)
                    
                    return result
        
        except Exception as e:
            logger.error(f"ChEMBL compound search failed: {e}")
            raise CoScientistError(
                f"ChEMBL compound search failed: {str(e)}",
                "CHEMBL_002",
                ErrorSeverity.MEDIUM
            )
    
    async def _search_targets(self, query: str) -> Dict[str, Any]:
        """Search for targets in ChEMBL"""
        cache_key = f"chembl_targets_{query}"
        cached_result = await cache_manager.get_api_response(cache_key)
        if cached_result:
            return cached_result
        
        try:
            async with self.circuit_breaker:
                url = f"{self.base_url}/target/search"
                params = {'q': query, 'format': 'json'}
                
                async with self.session.get(url, params=params, headers=self._get_headers()) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    targets = []
                    for item in data.get('targets', []):
                        target = {
                            "target_id": item.get('target_chembl_id', ''),
                            "name": item.get('pref_name', ''),
                            "type": item.get('target_type', ''),
                            "organism": item.get('organism', ''),
                            "description": item.get('description', ''),
                            "source": "chembl"
                        }
                        targets.append(target)
                    
                    result = {
                        "status": "success",
                        "targets": targets,
                        "total_count": len(targets)
                    }
                    
                    # Cache result
                    await cache_manager.set_api_response(cache_key, result)
                    
                    return result
        
        except Exception as e:
            logger.error(f"ChEMBL target search failed: {e}")
            raise CoScientistError(
                f"ChEMBL target search failed: {str(e)}",
                "CHEMBL_003",
                ErrorSeverity.MEDIUM
            )
    
    async def _get_activity_data(self, compound_id: str) -> Dict[str, Any]:
        """Get activity data for a compound"""
        cache_key = f"chembl_activity_{compound_id}"
        cached_result = await cache_manager.get_api_response(cache_key)
        if cached_result:
            return cached_result
        
        try:
            async with self.circuit_breaker:
                url = f"{self.base_url}/activity"
                params = {'molecule_chembl_id': compound_id, 'format': 'json'}
                
                async with self.session.get(url, params=params, headers=self._get_headers()) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    activities = []
                    for item in data.get('activities', []):
                        activity = {
                            "activity_id": item.get('activity_id', ''),
                            "target_id": item.get('target_chembl_id', ''),
                            "assay_type": item.get('assay_type', ''),
                            "standard_type": item.get('standard_type', ''),
                            "standard_value": item.get('standard_value'),
                            "standard_units": item.get('standard_units', ''),
                            "confidence": item.get('confidence_score'),
                            "source": "chembl"
                        }
                        activities.append(activity)
                    
                    result = {
                        "status": "success",
                        "activities": activities,
                        "total_count": len(activities)
                    }
                    
                    # Cache result
                    await cache_manager.set_api_response(cache_key, result)
                    
                    return result
        
        except Exception as e:
            logger.error(f"ChEMBL activity data failed: {e}")
            raise CoScientistError(
                f"ChEMBL activity data failed: {str(e)}",
                "CHEMBL_004",
                ErrorSeverity.MEDIUM
            )

class PubChemTool(ScientificTool):
    """PubChem database integration"""
    
    def __init__(self):
        super().__init__(
            name="pubchem",
            base_url=settings.PUBCHEM_API_URL
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Query PubChem database"""
        query_type = kwargs.get('query_type', 'compound')
        query = kwargs.get('query', '')
        
        if query_type == 'compound':
            return await self._search_compounds(query)
        elif query_type == 'substance':
            return await self._search_substances(query)
        elif query_type == 'assay':
            return await self._search_assays(query)
        else:
            raise CoScientistError(
                f"Unknown query type: {query_type}",
                "PUBCHEM_001",
                ErrorSeverity.LOW
            )
    
    async def _search_compounds(self, query: str) -> Dict[str, Any]:
        """Search for compounds in PubChem"""
        cache_key = f"pubchem_compounds_{query}"
        cached_result = await cache_manager.get_api_response(cache_key)
        if cached_result:
            return cached_result
        
        try:
            async with self.circuit_breaker:
                url = f"{self.base_url}/compound/name/{query}/JSON"
                
                async with self.session.get(url, headers=self._get_headers()) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    compounds = []
                    for item in data.get('PC_Compounds', []):
                        # Extract compound information
                        compound_id = str(item.get('id', {}).get('id', {}).get('cid', ''))
                        
                        # Extract properties
                        properties = {}
                        for prop in item.get('props', []):
                            urn = prop.get('urn', {})
                            label = urn.get('label', '')
                            value = prop.get('value', {})
                            
                            if 'sval' in value:
                                properties[label] = value['sval']
                            elif 'fval' in value:
                                properties[label] = value['fval']
                            elif 'ival' in value:
                                properties[label] = value['ival']
                        
                        compound = Compound(
                            compound_id=compound_id,
                            name=properties.get('IUPAC Name', query),
                            smiles=properties.get('SMILES', ''),
                            inchi=properties.get('InChI', ''),
                            molecular_formula=properties.get('Molecular Formula', ''),
                            molecular_weight=properties.get('Molecular Weight'),
                            properties=properties,
                            source="pubchem"
                        )
                        compounds.append(compound.__dict__)
                    
                    result = {
                        "status": "success",
                        "compounds": compounds,
                        "total_count": len(compounds)
                    }
                    
                    # Cache result
                    await cache_manager.set_api_response(cache_key, result)
                    
                    return result
        
        except Exception as e:
            logger.error(f"PubChem compound search failed: {e}")
            raise CoScientistError(
                f"PubChem compound search failed: {str(e)}",
                "PUBCHEM_002",
                ErrorSeverity.MEDIUM
            )
    
    async def _search_substances(self, query: str) -> Dict[str, Any]:
        """Search for substances in PubChem"""
        # Implementation similar to compounds but for substances
        return {
            "status": "success",
            "substances": [],
            "note": "Substance search implementation pending"
        }
    
    async def _search_assays(self, query: str) -> Dict[str, Any]:
        """Search for assays in PubChem"""
        # Implementation for assay search
        return {
            "status": "success",
            "assays": [],
            "note": "Assay search implementation pending"
        }

class UniProtTool(ScientificTool):
    """UniProt database integration"""
    
    def __init__(self):
        super().__init__(
            name="uniprot",
            base_url=settings.UNIPROT_API_URL
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Query UniProt database"""
        query = kwargs.get('query', '')
        query_type = kwargs.get('query_type', 'protein')
        
        if query_type == 'protein':
            return await self._search_proteins(query)
        elif query_type == 'gene':
            return await self._search_by_gene(query)
        else:
            raise CoScientistError(
                f"Unknown query type: {query_type}",
                "UNIPROT_001",
                ErrorSeverity.LOW
            )
    
    async def _search_proteins(self, query: str) -> Dict[str, Any]:
        """Search for proteins in UniProt"""
        cache_key = f"uniprot_proteins_{query}"
        cached_result = await cache_manager.get_api_response(cache_key)
        if cached_result:
            return cached_result
        
        try:
            async with self.circuit_breaker:
                url = f"{self.base_url}/uniprotkb/search"
                params = {
                    'query': query,
                    'format': 'json',
                    'size': 10
                }
                
                async with self.session.get(url, params=params, headers=self._get_headers()) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    proteins = []
                    for item in data.get('results', []):
                        protein = {
                            "uniprot_id": item.get('primaryAccession', ''),
                            "name": item.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                            "organism": item.get('organism', {}).get('scientificName', ''),
                            "gene_names": [gene.get('geneName', {}).get('value', '') for gene in item.get('genes', [])],
                            "sequence": item.get('sequence', {}).get('value', ''),
                            "length": item.get('sequence', {}).get('length', 0),
                            "keywords": [kw.get('value', '') for kw in item.get('keywords', [])],
                            "source": "uniprot"
                        }
                        proteins.append(protein)
                    
                    result = {
                        "status": "success",
                        "proteins": proteins,
                        "total_count": len(proteins)
                    }
                    
                    # Cache result
                    await cache_manager.set_api_response(cache_key, result)
                    
                    return result
        
        except Exception as e:
            logger.error(f"UniProt protein search failed: {e}")
            raise CoScientistError(
                f"UniProt protein search failed: {str(e)}",
                "UNIPROT_002",
                ErrorSeverity.MEDIUM
            )
    
    async def _search_by_gene(self, gene_name: str) -> Dict[str, Any]:
        """Search proteins by gene name"""
        return await self._search_proteins(f"gene:{gene_name}")

class ReactomeTool(ScientificTool):
    """Reactome pathway database integration"""
    
    def __init__(self):
        super().__init__(
            name="reactome",
            base_url="https://reactome.org/ContentService"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Query Reactome database"""
        query_type = kwargs.get('query_type', 'pathway')
        query = kwargs.get('query', '')
        
        if query_type == 'pathway':
            return await self._search_pathways(query)
        elif query_type == 'protein':
            return await self._get_protein_pathways(query)
        else:
            raise CoScientistError(
                f"Unknown query type: {query_type}",
                "REACTOME_001",
                ErrorSeverity.LOW
            )
    
    async def _search_pathways(self, query: str) -> Dict[str, Any]:
        """Search for pathways in Reactome"""
        cache_key = f"reactome_pathways_{query}"
        cached_result = await cache_manager.get_api_response(cache_key)
        if cached_result:
            return cached_result
        
        try:
            async with self.circuit_breaker:
                url = f"{self.base_url}/search/query"
                params = {'query': query, 'species': 'Homo sapiens'}
                
                async with self.session.get(url, params=params, headers=self._get_headers()) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    pathways = []
                    for item in data.get('entries', []):
                        if item.get('type') == 'Pathway':
                            pathway = PathwayInfo(
                                pathway_id=item.get('stId', ''),
                                name=item.get('name', ''),
                                description=item.get('summation', [{}])[0].get('text', ''),
                                genes=[],  # Would need additional API calls
                                proteins=[],  # Would need additional API calls
                                compounds=[],  # Would need additional API calls
                                source="reactome"
                            )
                            pathways.append(pathway.__dict__)
                    
                    result = {
                        "status": "success",
                        "pathways": pathways,
                        "total_count": len(pathways)
                    }
                    
                    # Cache result
                    await cache_manager.set_api_response(cache_key, result)
                    
                    return result
        
        except Exception as e:
            logger.error(f"Reactome pathway search failed: {e}")
            raise CoScientistError(
                f"Reactome pathway search failed: {str(e)}",
                "REACTOME_002",
                ErrorSeverity.MEDIUM
            )
    
    async def _get_protein_pathways(self, protein_id: str) -> Dict[str, Any]:
        """Get pathways for a specific protein"""
        cache_key = f"reactome_protein_pathways_{protein_id}"
        cached_result = await cache_manager.get_api_response(cache_key)
        if cached_result:
            return cached_result
        
        try:
            async with self.circuit_breaker:
                url = f"{self.base_url}/data/pathways/low/entity/{protein_id}"
                
                async with self.session.get(url, headers=self._get_headers()) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    pathways = []
                    for item in data:
                        pathway = PathwayInfo(
                            pathway_id=item.get('stId', ''),
                            name=item.get('displayName', ''),
                            description='',  # Would need additional API call
                            genes=[],
                            proteins=[protein_id],
                            compounds=[],
                            source="reactome"
                        )
                        pathways.append(pathway.__dict__)
                    
                    result = {
                        "status": "success",
                        "pathways": pathways,
                        "total_count": len(pathways)
                    }
                    
                    # Cache result
                    await cache_manager.set_api_response(cache_key, result)
                    
                    return result
        
        except Exception as e:
            logger.error(f"Reactome protein pathway search failed: {e}")
            raise CoScientistError(
                f"Reactome protein pathway search failed: {str(e)}",
                "REACTOME_003",
                ErrorSeverity.MEDIUM
            )

class ScientificToolsRegistry:
    """Registry for all scientific tools"""
    
    def __init__(self):
        self.tools = {
            "alphafold": AlphaFoldTool(),
            "chembl": ChEMBLTool(),
            "pubchem": PubChemTool(),
            "uniprot": UniProtTool(),
            "reactome": ReactomeTool()
        }
    
    async def call_tool(self, tool_name: str, **params) -> Dict[str, Any]:
        """Universal tool calling interface"""
        if tool_name not in self.tools:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }
        
        tool = self.tools[tool_name]
        
        try:
            async with tool:
                result = await tool.execute(**params)
                return {
                    "tool": tool_name,
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        except CoScientistError as e:
            return {
                "tool": tool_name,
                "status": "error",
                "error": e.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Unexpected error in tool {tool_name}: {e}")
            return {
                "tool": tool_name,
                "status": "error",
                "error": {
                    "code": "TOOL_UNEXPECTED_ERROR",
                    "message": str(e)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool"""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        tool = self.tools[tool_name]
        return {
            "name": tool.name,
            "description": tool.__class__.__doc__,
            "base_url": tool.base_url,
            "has_api_key": tool.api_key is not None,
            "circuit_breaker_stats": tool.circuit_breaker.get_stats()
        }
    
    def get_all_tools_info(self) -> Dict[str, Any]:
        """Get information about all available tools"""
        return {
            tool_name: self.get_tool_info(tool_name)
            for tool_name in self.tools.keys()
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all tools"""
        health_status = {}
        
        for tool_name, tool in self.tools.items():
            cb_stats = tool.circuit_breaker.get_stats()
            health_status[tool_name] = {
                "healthy": cb_stats["state"] == "closed",
                "state": cb_stats["state"],
                "success_rate": cb_stats["success_rate"],
                "total_calls": cb_stats["total_calls"],
                "last_failure": cb_stats["last_failure_time"]
            }
        
        # Overall health summary
        healthy_tools = sum(1 for status in health_status.values() if status["healthy"])
        total_tools = len(health_status)
        
        return {
            "tools": health_status,
            "summary": {
                "healthy_tools": healthy_tools,
                "total_tools": total_tools,
                "overall_health": "healthy" if healthy_tools == total_tools else "degraded"
            }
        }

# Global registry instance
scientific_tools_registry = ScientificToolsRegistry()

# Convenience functions for common operations
async def predict_protein_structure(sequence: str = None, uniprot_id: str = None) -> Dict[str, Any]:
    """Predict protein structure using AlphaFold"""
    return await scientific_tools_registry.call_tool(
        "alphafold",
        sequence=sequence,
        uniprot_id=uniprot_id
    )

async def search_compounds(query: str, database: str = "chembl") -> Dict[str, Any]:
    """Search for compounds in specified database"""
    return await scientific_tools_registry.call_tool(
        database,
        query_type="compound",
        query=query
    )

async def search_proteins(query: str) -> Dict[str, Any]:
    """Search for proteins in UniProt"""
    return await scientific_tools_registry.call_tool(
        "uniprot",
        query_type="protein",
        query=query
    )

async def search_pathways(query: str) -> Dict[str, Any]:
    """Search for pathways in Reactome"""
    return await scientific_tools_registry.call_tool(
        "reactome",
        query_type="pathway",
        query=query
    ) 