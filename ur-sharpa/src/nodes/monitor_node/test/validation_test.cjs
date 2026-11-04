const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function harness() {
  const source = fs.readFileSync(path.join(__dirname, '../monitor_node/static/index.html'), 'utf8');
  const begin = source.indexOf('    function renderValidation(');
  const end = source.indexOf('\n    function ', begin + 1);
  function element() {
    return {children: [], hidden: false, append(...values) { this.children.push(...values); },
      replaceChildren(...values) { this.children = values; }};
  }
  const nodes = Object.fromEntries(['record-validation', 'validation-grid', 'validation-summary'].map(k => [k, element()]));
  const context = {byId: key => nodes[key], finite: value => Number(value) || 0,
    document: {createElement: element}};
  vm.createContext(context);
  vm.runInContext(source.slice(begin, end), context);
  return {render: context.renderValidation, nodes};
}

test('validation renders enabled modalities including joint torque, and tactile-disabled episodes', () => {
  for (const count of [9, 7, 8]) {
    const {render, nodes} = harness();
    const items = Array.from({length: count}, () => ({ok: true, frames: 300, duration_s: 10, hz: 30}));
    render({items, ok: true, duration_s: 10}, false);
    assert.equal(nodes['record-validation'].hidden, false);
    assert.equal(nodes['validation-grid'].children.length, count);
  }
});

test('validation reports current rate threshold and detects stale repeated sources', () => {
  const {render, nodes} = harness();
  render({items: [{ok: false, rate_ok: false, minimum_hz: 29.4}]}, false);
  let state = nodes['validation-grid'].children[0].children[0].children[1];
  assert.equal(state.textContent, '<29.4 Hz');
  render({items: [{ok: false, rate_ok: true, timing: {stable: false}}]}, false);
  state = nodes['validation-grid'].children[0].children[0].children[1];
  assert.equal(state.textContent, '时序异常');
  render({items: [{ok: true}]}, true);
  assert.equal(nodes['record-validation'].hidden, true);
});
