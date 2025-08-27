import{E as y}from"./ExecutionDetailView-24ca68b8.js";import{m as x,g as h}from"./mockData-38ebf9b0.js";import"./createLucideIcon-a1714018.js";import"./index-8b3efc3f.js";import"./_commonjsHelpers-de833af9.js";import"./charts-bbe52c79.js";import"./clock-76184412.js";import"./zap-8b467b4b.js";import"./alert-circle-7b73ffd7.js";import"./chevron-right-a7178483.js";const S={title:"Components/ExecutionDetailView",component:y,parameters:{layout:"padded",docs:{description:{component:"Detailed view of a prompt execution with before/after comparison, expandable sections for prompt content, response, and metadata."}}},tags:["autodocs"]},e={args:{comparison:x.executionComparison}},r={args:{comparison:{current_execution:{execution_id:"exec_1705323456_abc123def",prompt_id:"code_generation_optimized",prompt_version:"2.1.0",conversation_id:"conv_user_session_456",prompt_text:"Create a TypeScript function that processes user authentication with JWT tokens, includes proper error handling, and follows best practices for security.",prompt_hash:"a1b2c3d4e5f67890",prompt_template:"code_generation",prompt_variables:{language:"typescript",complexity:"medium",security_level:"high"},model_name:"gpt-4-turbo-preview",model_version:"2024-03-01",model_parameters:{temperature:.2,max_tokens:2e3,top_p:.95},temperature:.2,max_tokens:2e3,timestamp:"2024-01-15T14:30:56Z",environment:{platform:"linux",python_version:"3.11.0",environment_type:"production"},git_commit:"a7b8c9d",context_length:1234,conversation_turn:1,response_text:`Here's a secure TypeScript function for JWT authentication:

\`\`\`typescript
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';

interface AuthPayload {
  userId: string;
  email: string;
  role: string;
}

export async function authenticateUser(
  email: string, 
  password: string,
  secretKey: string
): Promise<{ success: boolean; token?: string; error?: string }> {
  try {
    // Input validation
    if (!email || !password) {
      return { success: false, error: 'Email and password are required' };
    }

    // Here you would typically verify credentials against your database
    const user = await getUserByEmail(email);
    
    if (!user || !await bcrypt.compare(password, user.hashedPassword)) {
      return { success: false, error: 'Invalid credentials' };
    }

    // Create JWT payload
    const payload: AuthPayload = {
      userId: user.id,
      email: user.email,
      role: user.role
    };

    // Generate token with expiration
    const token = jwt.sign(payload, secretKey, { 
      expiresIn: '1h',
      issuer: 'your-app-name'
    });

    return { success: true, token };
  } catch (error) {
    return { 
      success: false, 
      error: 'Authentication service temporarily unavailable' 
    };
  }
}
\`\`\``,response_length:1456,response_tokens:364,execution_time_ms:234.5,tokens_per_second:12.4,memory_usage_mb:125.3,response_quality_score:.94,coherence_score:.96,relevance_score:.92,error_occurred:!1,error_message:void 0,error_type:void 0,baseline_execution_id:void 0,ab_test_group:void 0,experiment_tag:void 0},similar_executions:[{execution_id:"exec_1705320000_xyz789",prompt_version:"2.0.0",timestamp:"2024-01-15T13:45:30Z",execution_time_ms:312.1,response_length:1234,response_quality_score:.89,prompt_hash:"a1b2c3d4e5f67890"}],changes_detected:["Quality score improved by +0.05","Execution time improved by -24.8%","Response length increased by +18.0%"]}}},t={args:{comparison:{current_execution:{execution_id:"exec_1705323456_error123",prompt_id:"problematic_prompt_v1",prompt_version:"1.0.0",conversation_id:"conv_error_session",prompt_text:"Generate a complex algorithm that processes infinite data streams with undefined parameters.",prompt_hash:"error1234567890",prompt_template:"algorithm_generation",prompt_variables:{complexity:"infinite",data_type:"undefined"},model_name:"gpt-3.5-turbo",model_version:"2024-01-01",model_parameters:{temperature:1.8,max_tokens:500,top_p:1},temperature:1.8,max_tokens:500,timestamp:"2024-01-15T14:30:56Z",environment:{platform:"linux",python_version:"3.11.0",environment_type:"development"},git_commit:"error123",context_length:2456,conversation_turn:3,response_text:"",response_length:0,response_tokens:0,execution_time_ms:5e3,tokens_per_second:0,memory_usage_mb:89.2,response_quality_score:0,coherence_score:0,relevance_score:0,error_occurred:!0,error_message:"Request timed out after 5000ms. The model failed to generate a response within the specified time limit.",error_type:"timeout",baseline_execution_id:void 0,ab_test_group:"B",experiment_tag:"timeout_experiment"},similar_executions:[],changes_detected:["Error occurred - timeout after 5000ms","Temperature setting too high (1.8)","Prompt complexity may be causing issues"]}}},n={args:{comparison:h()}};var o,s,i;e.parameters={...e.parameters,docs:{...(o=e.parameters)==null?void 0:o.docs,source:{originalSource:`{
  args: {
    comparison: mockDataset.executionComparison
  }
}`,...(i=(s=e.parameters)==null?void 0:s.docs)==null?void 0:i.source}}};var a,p,c;r.parameters={...r.parameters,docs:{...(a=r.parameters)==null?void 0:a.docs,source:{originalSource:`{
  args: {
    comparison: {
      current_execution: {
        execution_id: 'exec_1705323456_abc123def',
        prompt_id: 'code_generation_optimized',
        prompt_version: '2.1.0',
        conversation_id: 'conv_user_session_456',
        prompt_text: 'Create a TypeScript function that processes user authentication with JWT tokens, includes proper error handling, and follows best practices for security.',
        prompt_hash: 'a1b2c3d4e5f67890',
        prompt_template: 'code_generation',
        prompt_variables: {
          language: 'typescript',
          complexity: 'medium',
          security_level: 'high'
        },
        model_name: 'gpt-4-turbo-preview',
        model_version: '2024-03-01',
        model_parameters: {
          temperature: 0.2,
          max_tokens: 2000,
          top_p: 0.95
        },
        temperature: 0.2,
        max_tokens: 2000,
        timestamp: '2024-01-15T14:30:56Z',
        environment: {
          platform: 'linux',
          python_version: '3.11.0',
          environment_type: 'production'
        },
        git_commit: 'a7b8c9d',
        context_length: 1234,
        conversation_turn: 1,
        response_text: \`Here's a secure TypeScript function for JWT authentication:

\\\`\\\`\\\`typescript
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';

interface AuthPayload {
  userId: string;
  email: string;
  role: string;
}

export async function authenticateUser(
  email: string, 
  password: string,
  secretKey: string
): Promise<{ success: boolean; token?: string; error?: string }> {
  try {
    // Input validation
    if (!email || !password) {
      return { success: false, error: 'Email and password are required' };
    }

    // Here you would typically verify credentials against your database
    const user = await getUserByEmail(email);
    
    if (!user || !await bcrypt.compare(password, user.hashedPassword)) {
      return { success: false, error: 'Invalid credentials' };
    }

    // Create JWT payload
    const payload: AuthPayload = {
      userId: user.id,
      email: user.email,
      role: user.role
    };

    // Generate token with expiration
    const token = jwt.sign(payload, secretKey, { 
      expiresIn: '1h',
      issuer: 'your-app-name'
    });

    return { success: true, token };
  } catch (error) {
    return { 
      success: false, 
      error: 'Authentication service temporarily unavailable' 
    };
  }
}
\\\`\\\`\\\`\`,
        response_length: 1456,
        response_tokens: 364,
        execution_time_ms: 234.5,
        tokens_per_second: 12.4,
        memory_usage_mb: 125.3,
        response_quality_score: 0.94,
        coherence_score: 0.96,
        relevance_score: 0.92,
        error_occurred: false,
        error_message: undefined,
        error_type: undefined,
        baseline_execution_id: undefined,
        ab_test_group: undefined,
        experiment_tag: undefined
      },
      similar_executions: [{
        execution_id: 'exec_1705320000_xyz789',
        prompt_version: '2.0.0',
        timestamp: '2024-01-15T13:45:30Z',
        execution_time_ms: 312.1,
        response_length: 1234,
        response_quality_score: 0.89,
        prompt_hash: 'a1b2c3d4e5f67890'
      }],
      changes_detected: ['Quality score improved by +0.05', 'Execution time improved by -24.8%', 'Response length increased by +18.0%']
    }
  }
}`,...(c=(p=r.parameters)==null?void 0:p.docs)==null?void 0:c.source}}};var m,_,u;t.parameters={...t.parameters,docs:{...(m=t.parameters)==null?void 0:m.docs,source:{originalSource:`{
  args: {
    comparison: {
      current_execution: {
        execution_id: 'exec_1705323456_error123',
        prompt_id: 'problematic_prompt_v1',
        prompt_version: '1.0.0',
        conversation_id: 'conv_error_session',
        prompt_text: 'Generate a complex algorithm that processes infinite data streams with undefined parameters.',
        prompt_hash: 'error1234567890',
        prompt_template: 'algorithm_generation',
        prompt_variables: {
          complexity: 'infinite',
          data_type: 'undefined'
        },
        model_name: 'gpt-3.5-turbo',
        model_version: '2024-01-01',
        model_parameters: {
          temperature: 1.8,
          max_tokens: 500,
          top_p: 1.0
        },
        temperature: 1.8,
        max_tokens: 500,
        timestamp: '2024-01-15T14:30:56Z',
        environment: {
          platform: 'linux',
          python_version: '3.11.0',
          environment_type: 'development'
        },
        git_commit: 'error123',
        context_length: 2456,
        conversation_turn: 3,
        response_text: '',
        response_length: 0,
        response_tokens: 0,
        execution_time_ms: 5000.0,
        tokens_per_second: 0,
        memory_usage_mb: 89.2,
        response_quality_score: 0,
        coherence_score: 0,
        relevance_score: 0,
        error_occurred: true,
        error_message: 'Request timed out after 5000ms. The model failed to generate a response within the specified time limit.',
        error_type: 'timeout',
        baseline_execution_id: undefined,
        ab_test_group: 'B',
        experiment_tag: 'timeout_experiment'
      },
      similar_executions: [],
      changes_detected: ['Error occurred - timeout after 5000ms', 'Temperature setting too high (1.8)', 'Prompt complexity may be causing issues']
    }
  }
}`,...(u=(_=t.parameters)==null?void 0:_.docs)==null?void 0:u.source}}};var d,l,g;n.parameters={...n.parameters,docs:{...(d=n.parameters)==null?void 0:d.docs,source:{originalSource:`{
  args: {
    comparison: generateMockExecutionComparison()
  }
}`,...(g=(l=n.parameters)==null?void 0:l.docs)==null?void 0:g.source}}};const C=["Default","SuccessfulExecution","ErrorExecution","ABTestExecution"];export{n as ABTestExecution,e as Default,t as ErrorExecution,r as SuccessfulExecution,C as __namedExportsOrder,S as default};
