import{c as x,j as e}from"./createLucideIcon-a1714018.js";import{D as T,A as S}from"./DashboardCard-4bfa3037.js";import{C as V,a as U}from"./clock-76184412.js";import{Z as W}from"./zap-8b467b4b.js";import"./index-8b3efc3f.js";import"./_commonjsHelpers-de833af9.js";import"./charts-bbe52c79.js";const D=x("Users",[["path",{d:"M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2",key:"1yyitq"}],["circle",{cx:"9",cy:"7",r:"4",key:"nufk8"}],["path",{d:"M22 21v-2a4 4 0 0 0-3-3.87",key:"kshegd"}],["path",{d:"M16 3.13a4 4 0 0 1 0 7.75",key:"1da9ce"}]]),q={title:"Components/DashboardCard",component:T,parameters:{layout:"padded",docs:{description:{component:"A metric card component for displaying key statistics with optional trend indicators."}}},tags:["autodocs"],argTypes:{trend:{control:"select",options:["up","down","stable"]}}},t={args:{title:"Total Executions",value:"15,432",subtitle:"All-time prompt executions",icon:e(S,{className:"h-8 w-8"})}},s={args:{title:"Success Rate",value:"96.8%",subtitle:"Successful executions",icon:e(V,{className:"h-8 w-8"}),trend:"up",trendValue:"+2.3% from last week"}},a={args:{title:"Average Response Time",value:"1.2s",subtitle:"Mean execution time",icon:e(W,{className:"h-8 w-8"}),trend:"down",trendValue:"+15ms slower than last week"}},r={args:{title:"Active Users",value:"127",subtitle:"Users in last 7 days",icon:e(D,{className:"h-8 w-8"}),trend:"stable",trendValue:"No significant change"}},n={args:{title:"Recent Activity",value:89,subtitle:"Last 24 hours",icon:e(U,{className:"h-8 w-8"}),trend:"up",trendValue:"Active"}},o={args:{title:"Custom Metric",value:"42",subtitle:"With custom styling",icon:e(S,{className:"h-8 w-8"}),className:"border-2 border-blue-200 shadow-lg"}};var c,i,l;t.parameters={...t.parameters,docs:{...(c=t.parameters)==null?void 0:c.docs,source:{originalSource:`{
  args: {
    title: 'Total Executions',
    value: '15,432',
    subtitle: 'All-time prompt executions',
    icon: <Activity className="h-8 w-8" />
  }
}`,...(l=(i=t.parameters)==null?void 0:i.docs)==null?void 0:l.source}}};var u,m,d;s.parameters={...s.parameters,docs:{...(u=s.parameters)==null?void 0:u.docs,source:{originalSource:`{
  args: {
    title: 'Success Rate',
    value: '96.8%',
    subtitle: 'Successful executions',
    icon: <CheckCircle className="h-8 w-8" />,
    trend: 'up',
    trendValue: '+2.3% from last week'
  }
}`,...(d=(m=s.parameters)==null?void 0:m.docs)==null?void 0:d.source}}};var p,h,g;a.parameters={...a.parameters,docs:{...(p=a.parameters)==null?void 0:p.docs,source:{originalSource:`{
  args: {
    title: 'Average Response Time',
    value: '1.2s',
    subtitle: 'Mean execution time',
    icon: <Zap className="h-8 w-8" />,
    trend: 'down',
    trendValue: '+15ms slower than last week'
  }
}`,...(g=(h=a.parameters)==null?void 0:h.docs)==null?void 0:g.source}}};var v,w,b;r.parameters={...r.parameters,docs:{...(v=r.parameters)==null?void 0:v.docs,source:{originalSource:`{
  args: {
    title: 'Active Users',
    value: '127',
    subtitle: 'Users in last 7 days',
    icon: <Users className="h-8 w-8" />,
    trend: 'stable',
    trendValue: 'No significant change'
  }
}`,...(b=(w=r.parameters)==null?void 0:w.docs)==null?void 0:b.source}}};var y,N,f;n.parameters={...n.parameters,docs:{...(y=n.parameters)==null?void 0:y.docs,source:{originalSource:`{
  args: {
    title: 'Recent Activity',
    value: 89,
    subtitle: 'Last 24 hours',
    icon: <Clock className="h-8 w-8" />,
    trend: 'up',
    trendValue: 'Active'
  }
}`,...(f=(N=n.parameters)==null?void 0:N.docs)==null?void 0:f.source}}};var k,A,C;o.parameters={...o.parameters,docs:{...(k=o.parameters)==null?void 0:k.docs,source:{originalSource:`{
  args: {
    title: 'Custom Metric',
    value: '42',
    subtitle: 'With custom styling',
    icon: <Activity className="h-8 w-8" />,
    className: 'border-2 border-blue-200 shadow-lg'
  }
}`,...(C=(A=o.parameters)==null?void 0:A.docs)==null?void 0:C.source}}};const H=["Default","WithUpTrend","WithDownTrend","WithStableTrend","NumericValue","CustomStyling"];export{o as CustomStyling,t as Default,n as NumericValue,a as WithDownTrend,r as WithStableTrend,s as WithUpTrend,H as __namedExportsOrder,q as default};
