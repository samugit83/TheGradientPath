"""
Code Generation Orchestrator - simplified direct execution approach
"""
import logging
from typing import Optional, Any
from context import MainContext, CodeGenContext, extract_usage_from_result
from .coder import create_coder, compile_code
from .tester import create_tester, compile_tests, run_tests
from .reviewer import create_reviewer, get_fallback_review
from agents import Runner

logger = logging.getLogger(__name__)


async def run_code_orchestrator(
    main_context: MainContext,
    max_iterations: int = 3,
    package_name: str = "app",
    max_tests: int = 15,
    constraints: Optional[dict] = None,
    session: Optional[Any] = None
) -> MainContext:
    """
    Direct execution implementation for code generation.
    This runs each agent sequentially and handles the logic directly.
    Now accepts MainContext and updates token usage.
    
    If session is provided, all sub-agents (coder, tester, reviewer) will
    share the same conversation history for better context and coherent workflow.
    """
    
    # Create or update the code generation context
    main_context.code_gen_context = CodeGenContext(
        constraints=constraints or {},
        package_name=package_name,
        max_iterations=max_iterations,
        max_tests=max_tests
    )
    
    context = main_context.code_gen_context
    
    coder = create_coder(main_context)
    tester = create_tester(main_context)
    reviewer = create_reviewer(main_context)
    
    for iteration in range(max_iterations):
        context.increment_iteration()
        logger.info(f"Starting iteration {context.iteration}")
        
        try:
            # Step 1: Generate code
            logger.info("Generating code...")
            # Prepare arguments for Runner.run
            run_kwargs = {
                "starting_agent": coder,
                "input": "Generate code based on the requirements", 
                "context": main_context
            }
            if session is not None:
                run_kwargs["session"] = session
            
            code_result = await Runner.run(**run_kwargs)
            
            # Update token usage
            usage = extract_usage_from_result(code_result)
            main_context.update_usage(usage)
            
            if code_result.final_output:
                try:
                    code_output = code_result.final_output 
                    compile_code(code_output.code, context.package_name)
                    context.update_code(code_output)
                    logger.info(f"Code generated: {code_output.explanation[:100]}...")
                except Exception as e:
                    logger.error(f"Failed to process code output: {e}")
                    continue
            
            # Step 2: Generate and run tests
            logger.info("Generating tests...")
            # Prepare arguments for Runner.run
            run_kwargs = {
                "starting_agent": tester,
                "input": "Generate comprehensive tests for the code",
                "context": main_context
            }
            if session is not None:
                run_kwargs["session"] = session
                
            test_result = await Runner.run(**run_kwargs)
            
            usage = extract_usage_from_result(test_result)
            main_context.update_usage(usage)
            
            if test_result.final_output:
                try:
                    test_output = test_result.final_output
                    
                    compile_tests(test_output.test_code, context.package_name)
                    test_results = run_tests(context.package_name)
                    test_results.test_explanation = test_output.explanation
                    context.test_results = test_results
                    
                    logger.info(f"Tests: {test_results.passed} passed, {test_results.failed} failed")
                except Exception as e:
                    logger.error(f"Failed to process test output: {e}")
                    continue
            
            # Step 3: Review results
            logger.info("Reviewing code and test results...")
            # Prepare arguments for Runner.run  
            run_kwargs = {
                "starting_agent": reviewer,
                "input": "Review the code and test results",
                "context": main_context
            }
            if session is not None:
                run_kwargs["session"] = session
                
            review_result = await Runner.run(**run_kwargs)
            
            # Update token usage
            usage = extract_usage_from_result(review_result)
            main_context.update_usage(usage)
            
            if review_result.final_output:
                try:
                    review_output = review_result.final_output 
                    
                    context.review_notes = review_output.review_notes
                    
                    if not review_output.should_continue or review_output.recommendation == "approve":
                        context.mark_complete()
                        logger.info("Process completed successfully!")
                        break
                    else:
                        logger.info(f"Review suggests revision: {review_output.review_notes[:2]}")
                        
                except Exception as e:
                    logger.error(f"Failed to process review output: {e}")
                    # Use fallback review
                    fallback = get_fallback_review(context)
                    context.review_notes = fallback.review_notes
                    if not fallback.should_continue:
                        context.mark_complete()
                        break
                        
        except Exception as e:
            logger.error(f"Error in iteration {context.iteration}: {e}")
            continue
    
    return main_context