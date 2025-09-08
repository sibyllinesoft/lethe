import{M as S}from"./ModelComparisonChart-5cefc860.js";import{m as f}from"./mockData-38ebf9b0.js";import"./createLucideIcon-a1714018.js";import"./index-8b3efc3f.js";import"./_commonjsHelpers-de833af9.js";import"./charts-bbe52c79.js";import"./generateCategoricalChart-ec1bcc95.js";import"./throttle-32ea0a72.js";import"./mapValues-fb986170.js";import"./tiny-invariant-dd7d57d2.js";import"./isPlainObject-cba39321.js";import"./_baseUniq-e6e71c30.js";import"./Line-c0585f89.js";import"./Scatter-7017bd04.js";const G={title:"Components/ModelComparisonChart",component:S,parameters:{layout:"padded",docs:{description:{component:"Combined bar and line chart comparing performance metrics across different AI models, showing execution time (bars) and quality scores (line)."}}},tags:["autodocs"],argTypes:{height:{control:{type:"range",min:300,max:800,step:50}}}},e={args:{data:f.modelComparison,height:400}},o={args:{data:[{model_name:"gpt-4-turbo-preview",execution_count:1456,avg_execution_time:234,avg_response_length:1234,avg_quality_score:.94,error_count:12},{model_name:"gpt-4",execution_count:987,avg_execution_time:456,avg_response_length:1456,avg_quality_score:.92,error_count:18},{model_name:"gpt-3.5-turbo",execution_count:2134,avg_execution_time:189,avg_response_length:890,avg_quality_score:.85,error_count:34}],height:400}},n={args:{data:[{model_name:"claude-3-opus-20240229",execution_count:567,avg_execution_time:345,avg_response_length:1567,avg_quality_score:.96,error_count:8},{model_name:"claude-3-sonnet-20240229",execution_count:892,avg_execution_time:278,avg_response_length:1234,avg_quality_score:.91,error_count:15},{model_name:"claude-3-haiku-20240307",execution_count:1234,avg_execution_time:156,avg_response_length:789,avg_quality_score:.87,error_count:23}],height:400}},t={args:{data:[{model_name:"high-speed-model",execution_count:1e3,avg_execution_time:89,avg_response_length:600,avg_quality_score:.75,error_count:45},{model_name:"balanced-model",execution_count:800,avg_execution_time:234,avg_response_length:1200,avg_quality_score:.89,error_count:12},{model_name:"high-quality-model",execution_count:500,avg_execution_time:567,avg_response_length:1800,avg_quality_score:.97,error_count:3},{model_name:"problematic-model",execution_count:200,avg_execution_time:2340,avg_response_length:1e3,avg_quality_score:.62,error_count:67}],height:500}},a={args:{data:[{model_name:"gpt-4-turbo-preview",execution_count:1456,avg_execution_time:234,avg_response_length:1234,avg_quality_score:.94,error_count:12}],height:300}},r={args:{data:f.modelComparison.slice(0,4),height:350}};var _,c,s;e.parameters={...e.parameters,docs:{...(_=e.parameters)==null?void 0:_.docs,source:{originalSource:`{
  args: {
    data: mockDataset.modelComparison,
    height: 400
  }
}`,...(s=(c=e.parameters)==null?void 0:c.docs)==null?void 0:s.source}}};var i,u,m;o.parameters={...o.parameters,docs:{...(i=o.parameters)==null?void 0:i.docs,source:{originalSource:`{
  args: {
    data: [{
      model_name: 'gpt-4-turbo-preview',
      execution_count: 1456,
      avg_execution_time: 234,
      avg_response_length: 1234,
      avg_quality_score: 0.94,
      error_count: 12
    }, {
      model_name: 'gpt-4',
      execution_count: 987,
      avg_execution_time: 456,
      avg_response_length: 1456,
      avg_quality_score: 0.92,
      error_count: 18
    }, {
      model_name: 'gpt-3.5-turbo',
      execution_count: 2134,
      avg_execution_time: 189,
      avg_response_length: 890,
      avg_quality_score: 0.85,
      error_count: 34
    }],
    height: 400
  }
}`,...(m=(u=o.parameters)==null?void 0:u.docs)==null?void 0:m.source}}};var g,l,p;n.parameters={...n.parameters,docs:{...(g=n.parameters)==null?void 0:g.docs,source:{originalSource:`{
  args: {
    data: [{
      model_name: 'claude-3-opus-20240229',
      execution_count: 567,
      avg_execution_time: 345,
      avg_response_length: 1567,
      avg_quality_score: 0.96,
      error_count: 8
    }, {
      model_name: 'claude-3-sonnet-20240229',
      execution_count: 892,
      avg_execution_time: 278,
      avg_response_length: 1234,
      avg_quality_score: 0.91,
      error_count: 15
    }, {
      model_name: 'claude-3-haiku-20240307',
      execution_count: 1234,
      avg_execution_time: 156,
      avg_response_length: 789,
      avg_quality_score: 0.87,
      error_count: 23
    }],
    height: 400
  }
}`,...(p=(l=n.parameters)==null?void 0:l.docs)==null?void 0:p.source}}};var d,v,h;t.parameters={...t.parameters,docs:{...(d=t.parameters)==null?void 0:d.docs,source:{originalSource:`{
  args: {
    data: [{
      model_name: 'high-speed-model',
      execution_count: 1000,
      avg_execution_time: 89,
      avg_response_length: 600,
      avg_quality_score: 0.75,
      error_count: 45
    }, {
      model_name: 'balanced-model',
      execution_count: 800,
      avg_execution_time: 234,
      avg_response_length: 1200,
      avg_quality_score: 0.89,
      error_count: 12
    }, {
      model_name: 'high-quality-model',
      execution_count: 500,
      avg_execution_time: 567,
      avg_response_length: 1800,
      avg_quality_score: 0.97,
      error_count: 3
    }, {
      model_name: 'problematic-model',
      execution_count: 200,
      avg_execution_time: 2340,
      avg_response_length: 1000,
      avg_quality_score: 0.62,
      error_count: 67
    }],
    height: 500
  }
}`,...(h=(v=t.parameters)==null?void 0:v.docs)==null?void 0:h.source}}};var x,y,q;a.parameters={...a.parameters,docs:{...(x=a.parameters)==null?void 0:x.docs,source:{originalSource:`{
  args: {
    data: [{
      model_name: 'gpt-4-turbo-preview',
      execution_count: 1456,
      avg_execution_time: 234,
      avg_response_length: 1234,
      avg_quality_score: 0.94,
      error_count: 12
    }],
    height: 300
  }
}`,...(q=(y=a.parameters)==null?void 0:y.docs)==null?void 0:q.source}}};var b,C,M;r.parameters={...r.parameters,docs:{...(b=r.parameters)==null?void 0:b.docs,source:{originalSource:`{
  args: {
    data: mockDataset.modelComparison.slice(0, 4),
    height: 350
  }
}`,...(M=(C=r.parameters)==null?void 0:C.docs)==null?void 0:M.source}}};const J=["Default","OpenAIModels","AnthropicModels","MixedPerformance","SingleModel","CompactHeight"];export{n as AnthropicModels,r as CompactHeight,e as Default,t as MixedPerformance,o as OpenAIModels,a as SingleModel,J as __namedExportsOrder,G as default};
