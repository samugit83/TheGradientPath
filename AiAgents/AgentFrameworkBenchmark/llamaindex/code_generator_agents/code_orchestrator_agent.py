"""
Code Generation Orchestrator - simplified direct execution approach
"""
import logging
from state import (
    MainState,
    CodeGenState,
    extract_usage_from_result,
    codegen_increment_iteration,
    codegen_update_code,
    codegen_mark_complete,
    update_main_usage,
)
from .coder import create_coder, compile_code
from .tester import create_tester, compile_tests, run_tests
from .reviewer import create_reviewer, get_fallback_review
from llama_index.core.workflow import Context
from .types import CodeOutput, TestCodeOutput, ReviewOutput
import json


logger = logging.getLogger(__name__)


async def run_code_orchestrator(ctx: Context) -> None:
    """
    Direct execution implementation for code generation.
    This runs each agent sequentially and handles the logic directly.
    Now accepts MainContext and updates token usage.
    
    If session is provided, all sub-agents (coder, tester, reviewer) will
    share the same conversation history for better context and coherent workflow.
    """
    
    async with ctx.store.edit_state() as ctx_state:
        mc: MainState = ctx_state["state"].get("main_state")
        if not mc:
            logger.error("MainState not found in context")
            return None
        
        existing_user_prompt = None
        if mc.code_gen_state:
            existing_user_prompt = mc.code_gen_state.user_prompt_for_app
        
        mc.code_gen_state = CodeGenState(user_prompt_for_app=existing_user_prompt)
        ctx_state["state"]["main_state"] = mc

    # Get the main state from state
    async with ctx.store.edit_state() as ctx_state:
        main_state: MainState = ctx_state["state"].get("main_state")
        if not main_state:
            logger.error("MainState not found in context")
            return None
        
        state = main_state.code_gen_state
        max_iterations = state.max_iterations

    for iteration in range(max_iterations):

        async with ctx.store.edit_state() as ctx_state:
            mc: MainState = ctx_state["state"].get("main_state")
            if not mc:
                logger.error("MainState not found in context")
                continue
            
            codegen_increment_iteration(mc.code_gen_state)
            ctx_state["state"]["main_state"] = mc
            state = mc.code_gen_state
            main_state = mc

        try:
            # Step 1: Generate code
            logger.info("Generating code...")
            
            # Create coder with current state for fresh instructions
            coder = create_coder(main_state)
            code_result = await coder.run("Generate code based on the requirements")
            
            usage = extract_usage_from_result(code_result, token_counter=main_state.token_counter)

            async with ctx.store.edit_state() as ctx_state:
                mc: MainState = ctx_state["state"].get("main_state")
                if not mc:
                    logger.error("MainState not found in context")
                    continue
                
                update_main_usage(mc, usage)
                ctx_state["state"]["main_state"] = mc
                main_state = mc

            # Parse the code result
            if hasattr(code_result, 'response') and code_result.response:
                try:
                    # Try to parse JSON response - extract JSON from response
                    response_text = str(code_result.response)
                    
                    # Find the JSON part (may have "assistant: " prefix)
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        code_data = json.loads(json_text)
                        code_output = CodeOutput(
                            code=code_data.get("code", ""),
                            explanation=code_data.get("explanation", "")
                        )
                    else:
                        # Fallback: treat as plain text
                        code_output = CodeOutput(
                            code=response_text,
                            explanation="Code generated from agent response"
                        )
                    
                    compile_code(code_output.code, state.package_name)

                    async with ctx.store.edit_state() as ctx_state:
                        mc: MainState = ctx_state["state"].get("main_state")
                        if not mc:
                            logger.error("MainState not found in context")
                            continue
                        
                        codegen_update_code(mc.code_gen_state, code_output)
                        ctx_state["state"]["main_state"] = mc
                        state = mc.code_gen_state

                    logger.info(f"Code generated: {code_output.explanation[:100]}...")
                except Exception as e:
                    logger.error(f"Failed to process code output: {e}")
                    continue
            
            # Step 2: Generate and run tests
            logger.info("Generating tests...")
            
            # Create tester with updated state (after code generation)
            tester = create_tester(main_state)
            test_result = await tester.run("Generate comprehensive tests for the code")
            
            usage = extract_usage_from_result(test_result, token_counter=main_state.token_counter)

            async with ctx.store.edit_state() as ctx_state:
                mc: MainState = ctx_state["state"].get("main_state")
                if not mc:
                    logger.error("MainState not found in context")
                    continue
                
                update_main_usage(mc, usage)
                ctx_state["state"]["main_state"] = mc
                main_state = mc
            
            # Parse the test result
            if hasattr(test_result, 'response') and test_result.response:
                try:
                    # Try to parse JSON response - extract JSON from response
                    response_text = str(test_result.response)
                    
                    # Find the JSON part (may have "assistant: " prefix)
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        test_data = json.loads(json_text)
                        test_output = TestCodeOutput(
                            test_code=test_data.get("test_code", ""),
                            explanation=test_data.get("explanation", ""),
                            test_count=test_data.get("test_count", 0)
                        )
                    else:
                        # Fallback: treat as plain text
                        test_output = TestCodeOutput(
                            test_code=response_text,
                            explanation="Test code generated from agent response",
                            test_count=1
                        )
                    
                    compile_tests(test_output.test_code, state.package_name)

                    test_results = run_tests(state.package_name)
                    test_results.test_explanation = test_output.explanation

                    async with ctx.store.edit_state() as ctx_state:
                        mc: MainState = ctx_state["state"].get("main_state")
                        if not mc:
                            logger.error("MainState not found in context")
                            continue
                        
                        mc.code_gen_state.test_results = test_results
                        ctx_state["state"]["main_state"] = mc
                        state = mc.code_gen_state

                    
                    logger.info(f"Tests: {test_results.passed} passed, {test_results.failed} failed")
                except Exception as e:
                    logger.error(f"Failed to process test output: {e}")
                    continue
            
            # Step 3: Review results
            logger.info("Reviewing code and test results...")
            
            # Create reviewer with updated state (after tests are run)
            reviewer = create_reviewer(main_state)
            review_result = await reviewer.run("Review the code and test results")
            
            # Update token usage
            usage = extract_usage_from_result(review_result, token_counter=main_state.token_counter)

            async with ctx.store.edit_state() as ctx_state:
                mc: MainState = ctx_state["state"].get("main_state")
                if not mc:
                    logger.error("MainState not found in context")
                    continue
                
                update_main_usage(mc, usage)
                ctx_state["state"]["main_state"] = mc
                main_state = mc


            if hasattr(review_result, 'response') and review_result.response:
                try:
                    # Try to parse JSON response - extract JSON from response
                    response_text = str(review_result.response)
                    
                    # Find the JSON part (may have "assistant: " prefix)
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        review_data = json.loads(json_text)
                        review_output = ReviewOutput(
                            review_notes=review_data.get("review_notes", []),
                            analysis=review_data.get("analysis", ""),
                            recommendation=review_data.get("recommendation", "revise"),
                            should_continue=review_data.get("recommendation", "revise") != "approve"
                        )
                    else:
                        # Fallback: treat as plain text
                        review_output = ReviewOutput(
                            review_notes=[response_text],
                            analysis="Review completed",
                            recommendation="revise",
                            should_continue=True
                        )
                    
                    async with ctx.store.edit_state() as ctx_state:
                        mc: MainState = ctx_state["state"].get("main_state")
                        if not mc:
                            logger.error("MainState not found in context")
                            continue
                        
                        mc.code_gen_state.review_notes = review_output.review_notes
                        ctx_state["state"]["main_state"] = mc
                        state = mc.code_gen_state

                    if not review_output.should_continue or review_output.recommendation == "approve":
                        async with ctx.store.edit_state() as ctx_state:
                            mc: MainState = ctx_state["state"].get("main_state")
                            if mc:
                                codegen_mark_complete(mc.code_gen_state)
                                ctx_state["state"]["main_state"] = mc
                        logger.info("Process completed successfully!")
                        break
                    else:
                        logger.info(f"Review suggests revision: {review_output.review_notes[:2]}")
                        
                except Exception as e:
                    logger.error(f"Failed to process review output: {e}")
                    # Use fallback review
                    fallback = get_fallback_review(main_state.code_gen_state)
                    
                    async with ctx.store.edit_state() as ctx_state:
                        mc: MainState = ctx_state["state"].get("main_state")
                        if mc:
                            mc.code_gen_state.review_notes = fallback.review_notes
                            if not fallback.should_continue:
                                codegen_mark_complete(mc.code_gen_state)
                                logger.info("Process completed with fallback review!")
                            ctx_state["state"]["main_state"] = mc
                    
                    if not fallback.should_continue:
                        break
                        
        except Exception as e:
            logger.error(f"Error in iteration {main_state.code_gen_state.iteration if main_state.code_gen_state else 'unknown'}: {e}")
            continue
    
    return None