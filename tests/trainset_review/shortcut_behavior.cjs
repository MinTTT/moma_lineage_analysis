// Execute the actual generated handler against a minimal DOM fixture.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
let count = 0;
const check = (condition) => { assert.ok(condition); count++; };
function element(tag='DIV', props={}) {
  return Object.assign({tagName:tag, hidden:false, parentElement:null, disabled:false,
    innerText:'', clicks:0, attrs:{}, rects:[{}], style:{display:'block',visibility:'visible'},
    getClientRects(){return this.rects;}, hasAttribute(key){return key in this.attrs;},
    getAttribute(key){return this.attrs[key] ?? null;}, closest(){return this.focusGuard ? this : null;},
    click(){this.clicks++;}}, props);
}
let buttons = [], dialogs = [], listeners = [];
const doc = {
  querySelectorAll(selector) { return selector === 'button' ? buttons : dialogs; },
  addEventListener(event, fn){listeners.push(fn);},
  removeEventListener(event, fn){listeners=listeners.filter(f=>f!==fn);},
};
const host={document:doc,getComputedStyle:n=>n.style};
const context=vm.createContext({window:{parent:host}});
const script=payload.script.trim().slice('<script>'.length, -'</script>'.length);
vm.runInContext(script,context);
const fire=(key, options={})=>{
  const event=Object.assign({key,target:element('BODY'),defaultPrevented:false,isComposing:false,
    repeat:false,ctrlKey:false,metaKey:false,altKey:false,
    preventDefault(){this.defaultPrevented=true;}},options);
  for(const handler of listeners)handler(event);
  return event;
};
const labels=payload.labels;
for(const [key,label] of Object.entries(labels)) {
  const b=element('BUTTON',{innerText:label}); buttons=[b];dialogs=[];
  const event=fire(key);
  check(b.clicks===1 && event.defaultPrevented);
  if(key.length===1){fire(key.toUpperCase());check(b.clicks===2);}
}
const b=element('BUTTON',{innerText:labels.d});buttons=[b];
for(const key of ['ctrlKey','metaKey','altKey','repeat','isComposing','defaultPrevented']) {
  const before=b.clicks;fire('d',{[key]:true});check(b.clicks===before);
}
for(const tag of ['INPUT','TEXTAREA','SELECT']) {
  const before=b.clicks;fire('d',{target:element(tag)});check(b.clicks===before);
}
for(const target of [element('DIV',{isContentEditable:true}),element('SPAN',{focusGuard:true})]) {
  const before=b.clicks;fire('d',{target});check(b.clicks===before);
}
// No hidden, disabled, collapsed, or inert action can be invoked.
for(const properties of [{disabled:true},{hidden:true},{rects:[]},
                         {attrs:{'aria-hidden':'true'}},{attrs:{inert:''}},
                         {style:{display:'none',visibility:'visible'}},
                         {style:{display:'block',visibility:'hidden'}}]) {
  const blocked=element('BUTTON',{innerText:labels.d,...properties});buttons=[blocked];
  const event=fire('d');check(blocked.clicks===0 && !event.defaultPrevented);
}
for(const parent of [element('DETAILS',{open:false}),element('DIV',{hidden:true})]) {
  const blocked=element('BUTTON',{innerText:labels.d,parentElement:parent});buttons=[blocked];
  fire('d');check(blocked.clicks===0);
}
buttons=[b];dialogs=[element('DIV')];const before=b.clicks;fire('d');check(b.clicks===before);
dialogs=[element('DIV',{hidden:true})];fire('d');check(b.clicks===before+1);
dialogs=[];
// A hidden duplicate must not mask the current visible button.
const hidden=element('BUTTON',{innerText:labels.d,hidden:true});buttons=[hidden,b];fire('d');
check(hidden.clicks===0 && b.clicks===before+2);
vm.runInContext(script,context);check(listeners.length===1);
const prior=b.clicks;fire('d');check(b.clicks===prior+1);
const unknown=fire('not-bound');check(!unknown.defaultPrevented);
console.log(JSON.stringify({checks:count,status:'PASS'}));
