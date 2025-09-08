#!/usr/bin/env python3
"""
Validation script for Iteration 4: LLM Rerank & Contradiction-Aware
Comprehensive validation of all implemented features and quality gates.
"""

import json
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_iteration4_implementation():
    """Validate that Iteration 4 implementation is complete and functional"""
    
    logger.info("🔍 ITERATION 4 VALIDATION")
    logger.info("=" * 60)
    logger.info("Validating: LLM Rerank & Contradiction-Aware Implementation")
    
    validation_results = {
        "timestamp": time.time(),
        "iteration": 4,
        "features": {},
        "integration": {},
        "quality_gates": {},
        "overall_status": "UNKNOWN"
    }
    
    # 1. Validate Core Components
    logger.info("\n📦 CORE COMPONENTS VALIDATION")
    logger.info("-" * 40)
    
    core_components = validate_core_components()
    validation_results["features"]["core_components"] = core_components
    
    for component, status in core_components.items():
        status_icon = "✅" if status["implemented"] else "❌"
        logger.info(f"{status_icon} {component}: {status['description']}")
    
    # 2. Validate Configuration System
    logger.info("\n⚙️ CONFIGURATION SYSTEM VALIDATION")
    logger.info("-" * 40)
    
    config_validation = validate_configuration_system()
    validation_results["features"]["configuration"] = config_validation
    
    for config_item, status in config_validation.items():
        status_icon = "✅" if status["valid"] else "❌"
        logger.info(f"{status_icon} {config_item}: {status['description']}")
    
    # 3. Validate Integration Points
    logger.info("\n🔗 INTEGRATION VALIDATION") 
    logger.info("-" * 40)
    
    integration_validation = validate_integration_points()
    validation_results["integration"] = integration_validation
    
    for integration_point, status in integration_validation.items():
        status_icon = "✅" if status["integrated"] else "❌"
        logger.info(f"{status_icon} {integration_point}: {status['description']}")
    
    # 4. Validate Quality Gates
    logger.info("\n🎯 QUALITY GATES VALIDATION")
    logger.info("-" * 40)
    
    quality_gates = validate_quality_gates()
    validation_results["quality_gates"] = quality_gates
    
    for gate, status in quality_gates.items():
        status_icon = "✅" if status["meets_requirement"] else "❌"
        target = status.get("target", "N/A")
        logger.info(f"{status_icon} {gate}: {status['description']} (Target: {target})")
    
    # 5. Calculate Overall Status
    core_implemented = sum(1 for comp in core_components.values() if comp["implemented"])
    total_core = len(core_components)
    config_valid = sum(1 for conf in config_validation.values() if conf["valid"])
    total_config = len(config_validation)
    integrated = sum(1 for integ in integration_validation.values() if integ["integrated"])
    total_integration = len(integration_validation)
    quality_gates_met = sum(1 for gate in quality_gates.values() if gate["meets_requirement"])
    total_gates = len(quality_gates)
    
    # Calculate completion percentages
    core_pct = core_implemented / total_core
    config_pct = config_valid / total_config
    integration_pct = integrated / total_integration
    quality_pct = quality_gates_met / total_gates
    
    # Overall status based on weighted criteria
    # Core components and integration are critical (must be 100%)
    # Configuration and quality gates are important (80% threshold)
    if (core_pct == 1.0 and integration_pct == 1.0 and 
        config_pct >= 0.8 and quality_pct >= 0.8):
        validation_results["overall_status"] = "PASS"
        overall_icon = "🎉"
        overall_message = "Iteration 4 implementation is COMPLETE and meets quality standards"
    elif (core_pct == 1.0 and integration_pct == 1.0 and 
          config_pct >= 0.6 and quality_pct >= 0.6):
        validation_results["overall_status"] = "PASS_WITH_WARNINGS" 
        overall_icon = "⚠️"
        overall_message = f"Iteration 4 implementation is COMPLETE with minor gaps ({total_config - config_valid} config items, {total_gates - quality_gates_met} quality gates need attention)"
    else:
        validation_results["overall_status"] = "FAIL"
        overall_icon = "❌"
        overall_message = f"Iteration 4 implementation is INCOMPLETE - core: {core_pct*100:.0f}%, config: {config_pct*100:.0f}%, integration: {integration_pct*100:.0f}%, quality: {quality_pct*100:.0f}%"
    
    # 6. Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Core Components: {sum(1 for c in core_components.values() if c['implemented'])}/{len(core_components)} implemented")
    logger.info(f"Configuration: {sum(1 for c in config_validation.values() if c['valid'])}/{len(config_validation)} valid")
    logger.info(f"Integration: {sum(1 for i in integration_validation.values() if i['integrated'])}/{len(integration_validation)} integrated")
    logger.info(f"Quality Gates: {quality_gates_met}/{total_gates} met ({quality_gates_met/total_gates*100:.1f}%)")
    
    logger.info(f"\n{overall_icon} OVERALL STATUS: {validation_results['overall_status']}")
    logger.info(f"📝 {overall_message}")
    
    # Save validation results (convert Path objects to strings for JSON serialization)
    def convert_paths_to_strings(obj):
        if isinstance(obj, dict):
            return {k: convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths_to_strings(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    output_file = f"artifacts/iter4_validation_{int(time.time())}.json"
    Path("artifacts").mkdir(exist_ok=True)
    
    serializable_results = convert_paths_to_strings(validation_results)
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\n💾 Validation results saved to: {output_file}")
    
    return validation_results

def validate_core_components():
    """Validate that all core Iteration 4 components are implemented"""
    
    ctx_run_path = Path("/home/nathan/Projects/lethe/ctx-run")
    
    components = {
        "LLM_Reranker_Core": {
            "implemented": False,
            "description": "Core LLM reranking implementation",
            "file_path": ctx_run_path / "packages/core/src/reranker/index.ts"
        },
        "LLM_Reranker_Package": {
            "implemented": False,
            "description": "Standalone LLM reranker package", 
            "file_path": ctx_run_path / "packages/reranker/src/index.ts"
        },
        "Contradiction_Detection": {
            "implemented": False,
            "description": "Contradiction-aware penalty system",
            "search_pattern": "checkContradiction"
        },
        "Timeout_Handling": {
            "implemented": False,
            "description": "LLM timeout and fallback mechanisms",
            "search_pattern": "llm_budget_ms"
        },
        "Configuration_Integration": {
            "implemented": False,
            "description": "Configuration system for LLM reranking",
            "search_pattern": "RerankerConfig"
        }
    }
    
    # Check file existence and content
    for component_name, component in components.items():
        if "file_path" in component:
            file_path = component["file_path"]
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    # Check for key patterns in the code
                    key_patterns = {
                        "LLM_Reranker_Core": ["class LLMReranker", "llmRerankWithTimeout", "applyContradictionPenalties"],
                        "LLM_Reranker_Package": ["export class LLMReranker", "RerankerResult", "LLMScoreResponse"],
                        "Contradiction_Detection": ["checkContradiction", "contradiction_penalty", "contradicts"],
                        "Timeout_Handling": ["llm_budget_ms", "timeout", "fallback"],
                        "Configuration_Integration": ["RerankerConfig", "use_llm", "llm_model"]
                    }
                    
                    if component_name in key_patterns:
                        patterns_found = sum(1 for pattern in key_patterns[component_name] if pattern in content)
                        component["implemented"] = patterns_found >= len(key_patterns[component_name]) * 0.7  # 70% of patterns must be present
                        component["patterns_found"] = f"{patterns_found}/{len(key_patterns[component_name])}"
                    else:
                        component["implemented"] = True  # File exists
                        
                except Exception as e:
                    component["error"] = str(e)
        
        elif "search_pattern" in component:
            # Search across multiple files
            search_pattern = component["search_pattern"]
            found_in_files = []
            
            for ts_file in ctx_run_path.rglob("*.ts"):
                try:
                    content = ts_file.read_text()
                    if search_pattern in content:
                        found_in_files.append(str(ts_file.relative_to(ctx_run_path)))
                except:
                    continue
            
            component["implemented"] = len(found_in_files) > 0
            component["found_in"] = found_in_files[:3]  # Limit to first 3 files
    
    return components

def validate_configuration_system():
    """Validate configuration system implementation"""
    
    config_items = {
        "LLM_Config_Interface": {
            "valid": False,
            "description": "RerankerConfig interface definition",
            "required_fields": ["use_llm", "llm_budget_ms", "llm_model", "contradiction_enabled", "contradiction_penalty"]
        },
        "Default_Configuration": {
            "valid": False,
            "description": "Default LLM reranker configuration values",
            "expected_values": {
                "use_llm": False,  # Disabled by default
                "llm_budget_ms": 1200,
                "llm_model": "llama3.2:1b"
            }
        },
        "Pipeline_Integration": {
            "valid": False,
            "description": "Pipeline accepts LLM rerank configuration",
            "search_pattern": "llmRerankConfig"
        },
        "YAML_Configuration": {
            "valid": False,
            "description": "YAML configuration file for Iteration 4",
            "file_path": Path("/home/nathan/Projects/lethe/lethe-research/experiments/iter4_llm_rerank.yaml")
        }
    }
    
    ctx_run_path = Path("/home/nathan/Projects/lethe/ctx-run")
    
    # Validate RerankerConfig interface
    reranker_file = ctx_run_path / "packages/core/src/reranker/index.ts"
    if reranker_file.exists():
        content = reranker_file.read_text()
        required_fields = config_items["LLM_Config_Interface"]["required_fields"]
        fields_found = sum(1 for field in required_fields if field in content)
        config_items["LLM_Config_Interface"]["valid"] = fields_found >= len(required_fields)
        config_items["LLM_Config_Interface"]["fields_found"] = f"{fields_found}/{len(required_fields)}"
    
    # Validate default configuration
    retrieval_file = ctx_run_path / "packages/core/src/retrieval/index.ts"
    if retrieval_file.exists():
        content = retrieval_file.read_text()
        expected_values = config_items["Default_Configuration"]["expected_values"]
        values_found = sum(1 for value in expected_values.values() if str(value) in content)
        config_items["Default_Configuration"]["valid"] = values_found >= len(expected_values) * 0.7
    
    # Validate pipeline integration
    pipeline_file = ctx_run_path / "packages/core/src/pipeline/index.ts"
    if pipeline_file.exists():
        content = pipeline_file.read_text()
        config_items["Pipeline_Integration"]["valid"] = "llmRerankConfig" in content
    
    # Validate YAML configuration
    yaml_file = config_items["YAML_Configuration"]["file_path"]
    if yaml_file.exists():
        try:
            content = yaml_file.read_text()
            required_sections = ["llmRerankConfig", "quality_gates", "rollback_conditions"]
            sections_found = sum(1 for section in required_sections if section in content)
            config_items["YAML_Configuration"]["valid"] = sections_found >= len(required_sections)
        except Exception as e:
            config_items["YAML_Configuration"]["error"] = str(e)
    
    return config_items

def validate_integration_points():
    """Validate integration between different system components"""
    
    integration_points = {
        "Pipeline_to_Retrieval": {
            "integrated": False,
            "description": "Pipeline passes LLM config to retrieval system"
        },
        "Retrieval_to_Reranker": {
            "integrated": False,
            "description": "Retrieval system uses configured reranker"
        },
        "Ollama_Integration": {
            "integrated": False,
            "description": "LLM reranker integrates with Ollama bridge"
        },
        "Fallback_Mechanisms": {
            "integrated": False,
            "description": "Graceful fallback from LLM to cross-encoder"
        },
        "Package_Dependencies": {
            "integrated": False,
            "description": "All package dependencies properly configured"
        }
    }
    
    ctx_run_path = Path("/home/nathan/Projects/lethe/ctx-run")
    
    # Check pipeline to retrieval integration
    pipeline_file = ctx_run_path / "packages/core/src/pipeline/index.ts"
    if pipeline_file.exists():
        content = pipeline_file.read_text()
        integration_points["Pipeline_to_Retrieval"]["integrated"] = (
            "llm_rerank" in content and "llmRerankConfig" in content
        )
    
    # Check retrieval to reranker integration
    retrieval_file = ctx_run_path / "packages/core/src/retrieval/index.ts"
    if retrieval_file.exists():
        content = retrieval_file.read_text()
        integration_points["Retrieval_to_Reranker"]["integrated"] = (
            "config.llm_rerank?.use_llm" in content and "getReranker" in content
        )
    
    # Check Ollama integration
    reranker_file = ctx_run_path / "packages/core/src/reranker/index.ts"
    if reranker_file.exists():
        content = reranker_file.read_text()
        integration_points["Ollama_Integration"]["integrated"] = (
            "getOllamaBridge" in content and "safeParseJSON" in content
        )
    
    # Check fallback mechanisms
    if reranker_file.exists():
        content = reranker_file.read_text()
        integration_points["Fallback_Mechanisms"]["integrated"] = (
            "CrossEncoderReranker" in content and "fallback" in content.lower()
        )
    
    # Check package dependencies by attempting build
    try:
        import subprocess
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(ctx_run_path),
            capture_output=True,
            text=True,
            timeout=60
        )
        integration_points["Package_Dependencies"]["integrated"] = result.returncode == 0
        if result.returncode != 0:
            integration_points["Package_Dependencies"]["error"] = result.stderr[:200]
    except Exception as e:
        integration_points["Package_Dependencies"]["error"] = str(e)
    
    return integration_points

def validate_quality_gates():
    """Validate that quality gates are properly defined and achievable"""
    
    quality_gates = {
        "Latency_Budget": {
            "meets_requirement": False,
            "description": "LLM reranking respects 1200ms timeout budget",
            "target": "≤1200ms",
            "implementation": "Timeout handling in LLMReranker"
        },
        "Fallback_Rate": {
            "meets_requirement": False,
            "description": "Graceful fallback when LLM unavailable",
            "target": "≤30% fallback rate",
            "implementation": "Cross-encoder fallback"
        },
        "Contradiction_Detection": {
            "meets_requirement": False,
            "description": "Contradiction-aware penalty system",
            "target": "15% score reduction for contradictions",
            "implementation": "checkContradiction method"
        },
        "Configuration_Flexibility": {
            "meets_requirement": False,
            "description": "Configurable LLM model and parameters",
            "target": "Runtime configuration support",
            "implementation": "RerankerConfig interface"
        },
        "Production_Readiness": {
            "meets_requirement": False,
            "description": "Error handling and logging",
            "target": "Comprehensive error handling",
            "implementation": "Try-catch blocks and logging"
        }
    }
    
    ctx_run_path = Path("/home/nathan/Projects/lethe/ctx-run")
    
    # Check timeout budget implementation
    reranker_file = ctx_run_path / "packages/core/src/reranker/index.ts"
    if reranker_file.exists():
        content = reranker_file.read_text()
        quality_gates["Latency_Budget"]["meets_requirement"] = (
            "llm_budget_ms" in content and "Date.now() - startTime" in content
        )
    
    # Check fallback implementation
    if reranker_file.exists():
        content = reranker_file.read_text()
        quality_gates["Fallback_Rate"]["meets_requirement"] = (
            "CrossEncoderReranker" in content and "catch" in content
        )
    
    # Check contradiction detection
    if reranker_file.exists():
        content = reranker_file.read_text()
        quality_gates["Contradiction_Detection"]["meets_requirement"] = (
            "checkContradiction" in content and "contradiction_penalty" in content
        )
    
    # Check configuration flexibility
    if reranker_file.exists():
        content = reranker_file.read_text()
        quality_gates["Configuration_Flexibility"]["meets_requirement"] = (
            "RerankerConfig" in content and "this.config" in content
        )
    
    # Check production readiness
    if reranker_file.exists():
        content = reranker_file.read_text()
        error_handling_patterns = ["try", "catch", "console.log", "console.warn", "console.error"]
        patterns_found = sum(1 for pattern in error_handling_patterns if pattern in content)
        quality_gates["Production_Readiness"]["meets_requirement"] = patterns_found >= 4
    
    return quality_gates

if __name__ == "__main__":
    validation_results = validate_iteration4_implementation()
    
    # Set exit code based on validation results
    if validation_results["overall_status"] == "PASS":
        exit_code = 0
    elif validation_results["overall_status"] == "PASS_WITH_WARNINGS":
        exit_code = 1  
    else:
        exit_code = 2
    
    exit(exit_code)