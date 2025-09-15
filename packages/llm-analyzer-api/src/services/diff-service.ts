import { diff_match_patch } from 'diff-match-patch'
import * as jsondiffpatch from 'jsondiffpatch'
import { CallPair } from '../types.js'

export class DiffService {
  private dmp = new diff_match_patch()
  private jsonDiffer = jsondiffpatch.create({
    objectHash: (obj: any) => obj.id || obj.role || obj.timestamp,
    textDiff: {
      minLength: 60
    }
  })

  generateTextDiff(text1: string, text2: string) {
    const diffs = this.dmp.diff_main(text1, text2)
    this.dmp.diff_cleanupSemantic(diffs)
    
    return {
      text: this.dmp.diff_prettyHtml(diffs),
      raw: diffs,
      similarity: this.calculateSimilarity(diffs)
    }
  }

  generateJsonDiff(obj1: any, obj2: any) {
    const delta = this.jsonDiffer.diff(obj1, obj2)
    return {
      delta,
      hasChanges: !!delta,
      html: null // Remove HTML formatting for now
    }
  }

  generateCallComparison(callA: CallPair, callB: CallPair) {
    return {
      metadata: {
        callA: {
          id: callA.id,
          timestamp: callA.timestamp,
          provider: callA.provider,
          model: callA.model
        },
        callB: {
          id: callB.id,
          timestamp: callB.timestamp,
          provider: callB.provider,
          model: callB.model
        }
      },
      
      // Compare prompts
      prompt_diff: this.generateTextDiff(callA.prompt, callB.prompt),
      
      // Compare contexts
      context_diff: {
        pre: this.generateJsonDiff(callA.pre_context, callB.pre_context),
        post: this.generateJsonDiff(callA.post_context, callB.post_context)
      },
      
      // Compare parameters
      params_diff: this.generateJsonDiff(
        {
          model: callA.model,
          temperature: callA.temperature,
          max_tokens: callA.max_tokens
        },
        {
          model: callB.model,
          temperature: callB.temperature,
          max_tokens: callB.max_tokens
        }
      ),
      
      // Compare outputs if available
      output_diff: callA.completion && callB.completion 
        ? this.generateTextDiff(callA.completion, callB.completion)
        : null,
      
      // Compare performance metrics
      performance_diff: this.generateJsonDiff(
        {
          latency_ms: callA.latency_ms,
          input_tokens: callA.input_tokens,
          output_tokens: callA.output_tokens
        },
        {
          latency_ms: callB.latency_ms,
          input_tokens: callB.input_tokens,
          output_tokens: callB.output_tokens
        }
      ),
      
      // Compare transformations
      transform_diff: this.generateJsonDiff(
        callA.transform_changes,
        callB.transform_changes
      )
    }
  }

  generatePrePostComparison(call: CallPair) {
    if (!call.request) {
      return null
    }

    const request = call.request
    
    return {
      metadata: {
        call_id: call.id,
        timestamp: call.timestamp,
        provider: call.provider,
        model: call.model
      },
      
      // Compare pre/post transformation payload
      payload_diff: this.generateJsonDiff(
        request.pre_transform.payload,
        request.post_transform.payload
      ),
      
      // Compare contexts specifically
      context_diff: this.generateJsonDiff(
        call.pre_context,
        call.post_context
      ),
      
      // Size comparison
      size_diff: {
        pre_bytes: request.pre_transform.size_bytes,
        post_bytes: request.post_transform.size_bytes,
        change_percent: request.transform.size_change_percent,
        change_bytes: request.post_transform.size_bytes - request.pre_transform.size_bytes
      },
      
      // Token comparison
      token_diff: {
        pre_tokens: request.pre_transform.token_estimate,
        post_tokens: request.post_transform.token_estimate,
        change_tokens: request.post_transform.token_estimate - request.pre_transform.token_estimate
      },
      
      // Applied transformations
      transformations: request.transform.changes
    }
  }

  private calculateSimilarity(diffs: any[]): number {
    let match = 0
    let total = 0
    
    diffs.forEach(([op, text]: [number, string]) => {
      const length = text.length
      total += length
      if (op === 0) { // EQUAL
        match += length
      }
    })
    
    return total > 0 ? (match / total) * 100 : 100
  }
}