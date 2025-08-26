#!/usr/bin/env python3
"""
LetheBench-Agents Dataset Generator
==================================

Main orchestration system for generating the LetheBench-Agents dataset with
weak labeling for agent conversation evaluation. Integrates all components:
- Agent conversation simulation with realistic tool usage
- Weak labeling pipeline for minimal supporting atom identification  
- Privacy scrubbing with configurable redaction rules
- Deterministic generation with reproducible seeding
- Quality validation and export in evaluation-ready formats

Architecture:
- Deterministic seed-based generation for reproducibility
- Balanced sampling across scenarios and complexity levels
- Comprehensive atom logging with weak supervision labels
- Privacy-safe content with audit trails
- Schema-validated exports compatible with Milestones 1-4

Output Formats:
- atoms.jsonl: Individual conversation atoms with metadata
- labels.jsonl: Weak supervision labels for supporting evidence
- entities.jsonl: Extracted entities with confidence scores
- manifest.json: Dataset metadata, statistics, and provenance
"""

import json
import random
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import uuid
from collections import defaultdict, Counter
import argparse
import sys

# Import our components
from agent_harness import AgentConversationSimulator, AgentAtom, AgentScenarios
from weak_labeling import WeakLabeler, WeakLabel
from privacy_scrubber import PrivacyScrubber, ScrubberConfig, create_standard_scrubber
from builders.schema import DatasetManifest, QueryRecord, DomainType, ComplexityLevel

@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""
    # Generation parameters
    seed: int = 42
    total_queries: int = 1000
    min_queries_per_scenario: int = 200
    target_session_length_range: Tuple[int, int] = (10, 50)
    complexity_distribution: Dict[str, float] = None
    
    # Scenario configuration
    enabled_scenarios: List[str] = None
    scenario_weights: Dict[str, float] = None
    
    # Privacy settings
    privacy_level: str = "standard"  # minimal, standard, aggressive
    custom_scrubber_config: Optional[ScrubberConfig] = None
    
    # Quality settings
    min_atoms_per_session: int = 8
    max_atoms_per_session: int = 100
    min_weak_labels_per_session: int = 2
    
    # Export settings
    export_format: str = "jsonl"  # jsonl, json, both
    include_audit_trail: bool = True
    include_raw_content: bool = False  # For debugging only
    
    # Reproducibility settings
    deterministic_generation: bool = True
    hash_content: bool = True
    
    def __post_init__(self):
        if self.enabled_scenarios is None:
            self.enabled_scenarios = ['coding', 'data_wrangling', 'web_qa', 'cli_automation']
        
        if self.scenario_weights is None:
            # Equal weight by default
            self.scenario_weights = {scenario: 1.0 for scenario in self.enabled_scenarios}
        
        if self.complexity_distribution is None:
            self.complexity_distribution = {
                'simple': 0.3,
                'medium': 0.5,
                'complex': 0.2
            }

@dataclass
class GenerationStats:
    """Statistics for dataset generation process"""
    total_sessions_generated: int = 0
    total_atoms_generated: int = 0
    total_weak_labels_generated: int = 0
    sessions_by_scenario: Dict[str, int] = None
    atoms_by_type: Dict[str, int] = None
    privacy_scrubbing_stats: Dict[str, Any] = None
    generation_time_seconds: float = 0.0
    
    def __post_init__(self):
        if self.sessions_by_scenario is None:
            self.sessions_by_scenario = {}
        if self.atoms_by_type is None:
            self.atoms_by_type = {}

class LetheBenchAgentsGenerator:
    """Main dataset generation orchestrator"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components with consistent seeding
        self.simulator = AgentConversationSimulator(seed=config.seed)
        self.weak_labeler = WeakLabeler(seed=config.seed)
        self.privacy_scrubber = self._create_privacy_scrubber()
        
        # Generation state
        self.rng = random.Random(config.seed)
        self.generated_sessions = []
        self.all_atoms = []
        self.all_weak_labels = []
        self.generation_stats = GenerationStats()
        
        # Content hashing for integrity verification
        self.content_hasher = hashlib.sha256()
    
    def _create_privacy_scrubber(self) -> PrivacyScrubber:
        """Create privacy scrubber based on configuration"""
        if self.config.custom_scrubber_config:
            return PrivacyScrubber(self.config.custom_scrubber_config)
        elif self.config.privacy_level == "minimal":
            from privacy_scrubber import create_minimal_scrubber
            return create_minimal_scrubber()
        elif self.config.privacy_level == "aggressive":
            from privacy_scrubber import create_aggressive_scrubber
            return create_aggressive_scrubber()
        else:
            return create_standard_scrubber()
    
    def generate_dataset(self) -> Tuple[List[AgentAtom], List[WeakLabel], GenerationStats]:
        """Generate complete LetheBench-Agents dataset"""
        start_time = time.time()
        
        self.logger.info(f"Starting dataset generation with seed {self.config.seed}")
        self.logger.info(f"Target: {self.config.total_queries} queries across {len(self.config.enabled_scenarios)} scenarios")
        
        try:
            # Phase 1: Generate agent conversation sessions
            self._generate_conversation_sessions()
            
            # Phase 2: Apply privacy scrubbing
            self._apply_privacy_scrubbing()
            
            # Phase 3: Generate weak labels
            self._generate_weak_labels()
            
            # Phase 4: Apply quality filtering
            self._apply_quality_filtering()
            
            # Phase 5: Balance dataset composition
            self._balance_dataset()
            
            # Update generation statistics
            end_time = time.time()
            self.generation_stats.generation_time_seconds = end_time - start_time
            self.generation_stats.total_atoms_generated = len(self.all_atoms)
            self.generation_stats.total_weak_labels_generated = len(self.all_weak_labels)
            
            self.logger.info(f"Dataset generation completed in {self.generation_stats.generation_time_seconds:.2f} seconds")
            self.logger.info(f"Generated {self.generation_stats.total_atoms_generated} atoms with {self.generation_stats.total_weak_labels_generated} weak labels")
            
            return self.all_atoms, self.all_weak_labels, self.generation_stats
            
        except Exception as e:
            self.logger.error(f"Dataset generation failed: {str(e)}")
            raise
    
    def _generate_conversation_sessions(self):
        """Generate agent conversation sessions across all scenarios"""
        self.logger.info("Phase 1: Generating agent conversation sessions")
        
        # Calculate queries per scenario based on weights
        scenario_query_targets = self._calculate_scenario_targets()
        
        for scenario, target_queries in scenario_query_targets.items():
            self.logger.info(f"Generating {target_queries} queries for scenario: {scenario}")
            
            queries_generated = 0
            session_count = 0
            
            while queries_generated < target_queries:
                # Generate session length based on complexity distribution
                complexity = self._sample_complexity()
                session_length = self._calculate_session_length(complexity)
                
                # Generate conversation session
                try:
                    atoms = self.simulator.generate_session(
                        scenario_name=scenario,
                        target_turns=session_length,
                        complexity_level=complexity
                    )
                    
                    if len(atoms) >= self.config.min_atoms_per_session:
                        self.all_atoms.extend(atoms)
                        self.generated_sessions.append({
                            'session_id': atoms[0].session_id,
                            'scenario': scenario,
                            'complexity': complexity,
                            'atom_count': len(atoms),
                            'atoms': atoms
                        })
                        
                        # Count query atoms (user requests and agent responses)
                        query_atoms = [atom for atom in atoms if atom.atom_type in ['user_request', 'agent_response']]
                        queries_generated += len(query_atoms)
                        session_count += 1
                        
                        if session_count % 10 == 0:
                            self.logger.debug(f"  {scenario}: Generated {session_count} sessions, {queries_generated}/{target_queries} queries")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate session for {scenario}: {str(e)}")
                    continue
            
            self.generation_stats.sessions_by_scenario[scenario] = session_count
            self.logger.info(f"Completed {scenario}: {session_count} sessions, {queries_generated} queries")
        
        self.generation_stats.total_sessions_generated = len(self.generated_sessions)
        
        # Update atom type statistics
        atom_types = [atom.atom_type for atom in self.all_atoms]
        self.generation_stats.atoms_by_type = dict(Counter(atom_types))
    
    def _calculate_scenario_targets(self) -> Dict[str, int]:
        """Calculate target number of queries per scenario"""
        # Ensure minimum queries per scenario
        scenario_targets = {}
        total_weight = sum(self.config.scenario_weights.values())
        
        for scenario in self.config.enabled_scenarios:
            weight = self.config.scenario_weights[scenario]
            proportional_queries = int((weight / total_weight) * self.config.total_queries)
            scenario_targets[scenario] = max(proportional_queries, self.config.min_queries_per_scenario)
        
        return scenario_targets
    
    def _sample_complexity(self) -> str:
        """Sample complexity level based on distribution"""
        complexities = list(self.config.complexity_distribution.keys())
        weights = list(self.config.complexity_distribution.values())
        return self.rng.choices(complexities, weights=weights)[0]
    
    def _calculate_session_length(self, complexity: str) -> int:
        """Calculate session length based on complexity"""
        base_min, base_max = self.config.target_session_length_range
        
        complexity_multipliers = {
            'simple': 0.7,
            'medium': 1.0,
            'complex': 1.4
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        min_length = int(base_min * multiplier)
        max_length = int(base_max * multiplier)
        
        return self.rng.randint(min_length, max_length)
    
    def _apply_privacy_scrubbing(self):
        """Apply privacy scrubbing to all generated content"""
        self.logger.info("Phase 2: Applying privacy scrubbing")
        
        scrubbed_count = 0
        audit_entries = []
        
        for atom in self.all_atoms:
            scrubbed_content, atom_audit_entries = self.privacy_scrubber.scrub_content(
                atom.content, context=f"{atom.atom_type}_{atom.session_id}"
            )
            
            if scrubbed_content != atom.content:
                # Store original content if configured (debugging only)
                if self.config.include_raw_content:
                    atom.metadata['original_content'] = atom.content
                
                atom.content = scrubbed_content
                atom.metadata['privacy_scrubbed'] = True
                scrubbed_count += 1
            
            audit_entries.extend(atom_audit_entries)
        
        # Store privacy scrubbing statistics
        self.generation_stats.privacy_scrubbing_stats = {
            'atoms_scrubbed': scrubbed_count,
            'total_atoms': len(self.all_atoms),
            'scrubbing_rate': scrubbed_count / len(self.all_atoms) if self.all_atoms else 0,
            'audit_entries': len(audit_entries),
            'scrubber_stats': self.privacy_scrubber.get_scrubbing_stats()
        }
        
        self.logger.info(f"Privacy scrubbing completed: {scrubbed_count}/{len(self.all_atoms)} atoms modified")
    
    def _generate_weak_labels(self):
        """Generate weak supervision labels for all sessions"""
        self.logger.info("Phase 3: Generating weak supervision labels")
        
        for session_info in self.generated_sessions:
            try:
                session_atoms = session_info['atoms']
                weak_labels = self.weak_labeler.generate_weak_labels(session_atoms)
                self.all_weak_labels.extend(weak_labels)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate weak labels for session {session_info['session_id']}: {str(e)}")
        
        self.logger.info(f"Generated {len(self.all_weak_labels)} weak labels across {len(self.generated_sessions)} sessions")
    
    def _apply_quality_filtering(self):
        """Apply quality filters to remove low-quality sessions"""
        self.logger.info("Phase 4: Applying quality filtering")
        
        initial_sessions = len(self.generated_sessions)
        initial_atoms = len(self.all_atoms)
        initial_labels = len(self.all_weak_labels)
        
        # Filter sessions based on quality criteria
        quality_sessions = []
        quality_atoms = []
        quality_labels = []
        
        for session_info in self.generated_sessions:
            session_atoms = session_info['atoms']
            session_id = session_info['session_id']
            
            # Quality criteria
            has_sufficient_atoms = len(session_atoms) >= self.config.min_atoms_per_session
            has_tool_interactions = any(atom.atom_type in ['tool_action', 'tool_observation'] for atom in session_atoms)
            has_weak_labels = len([label for label in self.all_weak_labels if label.query_id.startswith(f"query_{session_id}")]) >= self.config.min_weak_labels_per_session
            
            if has_sufficient_atoms and has_tool_interactions and has_weak_labels:
                quality_sessions.append(session_info)
                quality_atoms.extend(session_atoms)
                
                # Keep weak labels for this session
                session_labels = [label for label in self.all_weak_labels 
                                if any(atom.atom_id in label.supporting_atom_ids for atom in session_atoms)]
                quality_labels.extend(session_labels)
        
        # Update with filtered data
        self.generated_sessions = quality_sessions
        self.all_atoms = quality_atoms
        self.all_weak_labels = quality_labels
        
        filtered_sessions = initial_sessions - len(quality_sessions)
        filtered_atoms = initial_atoms - len(quality_atoms)
        filtered_labels = initial_labels - len(quality_labels)
        
        self.logger.info(f"Quality filtering complete: removed {filtered_sessions} sessions, {filtered_atoms} atoms, {filtered_labels} labels")
    
    def _balance_dataset(self):
        """Balance dataset composition across scenarios and complexity levels"""
        self.logger.info("Phase 5: Balancing dataset composition")
        
        # Analyze current distribution
        scenario_distribution = defaultdict(list)
        complexity_distribution = defaultdict(list)
        
        for session_info in self.generated_sessions:
            scenario_distribution[session_info['scenario']].append(session_info)
            complexity_distribution[session_info['complexity']].append(session_info)
        
        self.logger.info("Current distribution:")
        for scenario, sessions in scenario_distribution.items():
            self.logger.info(f"  {scenario}: {len(sessions)} sessions")
        for complexity, sessions in complexity_distribution.items():
            self.logger.info(f"  {complexity}: {len(sessions)} sessions")
        
        # Apply any balancing logic here if needed
        # For now, we keep all quality sessions
        
        # Final statistics update
        self.generation_stats.sessions_by_scenario = {
            scenario: len(sessions) for scenario, sessions in scenario_distribution.items()
        }
    
    def export_dataset(self, output_dir: Path, dataset_name: str = "lethebench_agents") -> Path:
        """Export dataset in evaluation-ready format"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting dataset to {output_dir}")
        
        # Create dataset manifest
        manifest = self._create_dataset_manifest(dataset_name)
        
        # Export atoms
        atoms_path = output_dir / "atoms.jsonl"
        self._export_atoms(atoms_path)
        
        # Export weak labels  
        labels_path = output_dir / "labels.jsonl"
        self._export_weak_labels(labels_path)
        
        # Export entities
        entities_path = output_dir / "entities.jsonl"
        self._export_entities(entities_path)
        
        # Export manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(manifest), f, indent=2, ensure_ascii=False, default=str)
        
        # Export statistics
        stats_path = output_dir / "generation_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.generation_stats), f, indent=2, ensure_ascii=False, default=str)
        
        # Export privacy audit trail if configured
        if self.config.include_audit_trail:
            audit_path = output_dir / "privacy_audit.json"
            self.privacy_scrubber.export_audit_trail(str(audit_path))
        
        self.logger.info(f"Dataset export completed: {len(self.all_atoms)} atoms, {len(self.all_weak_labels)} labels")
        
        return output_dir
    
    def _create_dataset_manifest(self, dataset_name: str) -> Dict[str, Any]:
        """Create dataset manifest with metadata"""
        return {
            'dataset_name': dataset_name,
            'version': '1.0.0',
            'description': 'LetheBench-Agents: Agent conversation traces with weak supervision labels',
            'generation_config': asdict(self.config),
            'generation_stats': asdict(self.generation_stats),
            'schema_version': '1.0.0',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'total_atoms': len(self.all_atoms),
            'total_weak_labels': len(self.all_weak_labels),
            'total_sessions': len(self.generated_sessions),
            'scenarios': list(self.config.enabled_scenarios),
            'complexity_levels': list(self.config.complexity_distribution.keys()),
            'privacy_level': self.config.privacy_level,
            'reproducibility': {
                'seed': self.config.seed,
                'deterministic': self.config.deterministic_generation,
                'content_hash': self.content_hasher.hexdigest() if self.config.hash_content else None
            }
        }
    
    def _export_atoms(self, output_path: Path):
        """Export atoms to JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for atom in self.all_atoms:
                atom_dict = atom.to_dict()
                f.write(json.dumps(atom_dict, ensure_ascii=False) + '\n')
                
                # Update content hash
                if self.config.hash_content:
                    self.content_hasher.update(json.dumps(atom_dict, sort_keys=True).encode())
    
    def _export_weak_labels(self, output_path: Path):
        """Export weak labels to JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for label in self.all_weak_labels:
                label_dict = label.to_dict()
                f.write(json.dumps(label_dict, ensure_ascii=False) + '\n')
    
    def _export_entities(self, output_path: Path):
        """Export extracted entities to JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for atom in self.all_atoms:
                for entity in atom.entities:
                    entity_record = {
                        'atom_id': atom.atom_id,
                        'session_id': atom.session_id,
                        'entity_type': entity['type'],
                        'entity_value': entity['value'],
                        'confidence': entity['confidence'],
                        'context': atom.content[:200] + '...' if len(atom.content) > 200 else atom.content
                    }
                    f.write(json.dumps(entity_record, ensure_ascii=False) + '\n')

def main():
    """Command-line interface for dataset generation"""
    parser = argparse.ArgumentParser(description="Generate LetheBench-Agents dataset")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--total-queries', type=int, default=1000, help='Total number of queries to generate')
    parser.add_argument('--scenarios', nargs='+', default=['coding', 'data_wrangling', 'web_qa', 'cli_automation'],
                       help='Scenarios to include')
    parser.add_argument('--privacy-level', choices=['minimal', 'standard', 'aggressive'], default='standard',
                       help='Privacy scrubbing level')
    parser.add_argument('--output-dir', type=str, default='./lethebench_agents_output', 
                       help='Output directory for generated dataset')
    parser.add_argument('--dataset-name', type=str, default='lethebench_agents',
                       help='Name for the generated dataset')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create generation configuration
    config = GenerationConfig(
        seed=args.seed,
        total_queries=args.total_queries,
        enabled_scenarios=args.scenarios,
        privacy_level=args.privacy_level
    )
    
    try:
        # Generate dataset
        generator = LetheBenchAgentsGenerator(config)
        atoms, labels, stats = generator.generate_dataset()
        
        # Export dataset
        output_dir = generator.export_dataset(Path(args.output_dir), args.dataset_name)
        
        print(f"\n✅ Dataset generation completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Generated {len(atoms)} atoms with {len(labels)} weak labels")
        print(f"⏱️  Generation time: {stats.generation_time_seconds:.2f} seconds")
        print(f"🎯 Scenarios: {', '.join(stats.sessions_by_scenario.keys())}")
        print(f"🔒 Privacy level: {config.privacy_level}")
        
    except Exception as e:
        print(f"\n❌ Dataset generation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()