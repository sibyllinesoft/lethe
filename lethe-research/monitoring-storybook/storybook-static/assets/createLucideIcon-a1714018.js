import{r as i}from"./index-8b3efc3f.js";var l={exports:{}},p={};/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var y=i,v=Symbol.for("react.element"),w=Symbol.for("react.fragment"),j=Object.prototype.hasOwnProperty,k=y.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,E={key:!0,ref:!0,__self:!0,__source:!0};function c(t,e,n){var r,o={},s=null,a=null;n!==void 0&&(s=""+n),e.key!==void 0&&(s=""+e.key),e.ref!==void 0&&(a=e.ref);for(r in e)j.call(e,r)&&!E.hasOwnProperty(r)&&(o[r]=e[r]);if(t&&t.defaultProps)for(r in e=t.defaultProps,e)o[r]===void 0&&(o[r]=e[r]);return{$$typeof:v,type:t,key:s,ref:a,props:o,_owner:k.current}}p.Fragment=w;p.jsx=c;p.jsxs=c;l.exports=p;var f=l.exports;const O=f.jsx,g=f.jsxs;var b={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};const h=t=>t.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase(),C=(t,e)=>{const n=i.forwardRef(({color:r="currentColor",size:o=24,strokeWidth:s=2,absoluteStrokeWidth:a,children:u,...m},_)=>i.createElement("svg",{ref:_,...b,width:o,height:o,stroke:r,strokeWidth:a?Number(s)*24/Number(o):s,className:`lucide lucide-${h(t)}`,...m},[...e.map(([d,x])=>i.createElement(d,x)),...(Array.isArray(u)?u:[u])||[]]));return n.displayName=`${t}`,n};export{g as a,C as c,O as j};
