// 실제 시간으로 헤드리스 Chrome 을 띄워 뷰어를 조작하고 스크린샷을 찍는다.
// 사용: node cdp.mjs <url> <outdir> <steps.json>
// steps: [{throttle:bytesPerSec} | {navigate:url} | {wait:ms} | {shot:name} | {move:[x,y]} | {click:[x,y]} | {moveTile:i} | {clickTile:i} | {key:"ArrowUp", code?, shift?} | {type:"text"} | {clickSel:"css"} | {burst:"name", n} | {eval:"js"}]
// click/clickSel 다음 단계에 burst 를 두면 전환 중 프레임을 연달아 찍는다
import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [url, outdir, stepsFile] = process.argv.slice(2);
const steps = JSON.parse(readFileSync(stepsFile, 'utf8'));
const port = 9333 + Math.floor(Math.random() * 300);
const prof = mkdtempSync(join(tmpdir(), 'cdp-'));
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${prof}`, '--allow-file-access-from-files',
  '--window-size=1600,1000', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));

let targets;
for (let k = 0; k < 50; k++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); if (targets.length) break; } catch {}
  await sleep(200);
}
const page = targets.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let id = 0; const wait = {};
const logs = [];
ws.addEventListener('message', ev => {
  const m = JSON.parse(ev.data);
  if (m.id && wait[m.id]) { wait[m.id](m); delete wait[m.id]; }
  if (m.method === 'Runtime.exceptionThrown') logs.push('EXC ' + JSON.stringify(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') logs.push('ERR ' + m.params.args.map(a => a.value).join(' '));
});
const send = (method, params = {}) => new Promise(r => { const i = ++id; wait[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
await send('Runtime.enable'); await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url });
await sleep(1500);
for (const s of steps) {
  if (s.throttle) {   // 느린 회선 흉내: {throttle: 초당 바이트}. 그 다음 {navigate: url} 로 다시 연다
    await send('Network.enable');
    await send('Network.emulateNetworkConditions', { offline: false, latency: 80, downloadThroughput: s.throttle, uploadThroughput: s.throttle });
  }
  if (s.viewport) {   // {viewport: [너비, 높이, 폰 여부]}
    const [width, height, mobile = false] = s.viewport;
    await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    if (mobile) await send('Emulation.setTouchEmulationEnabled', { enabled: true });
  }
  if (s.drag) {   // {drag: [x0, y0, x1, y1]} 마우스로 끌기
    const [x0, y0, x1, y1] = s.drag;
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0, y: y0 });
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: x0, y: y0, button: 'left', clickCount: 1 });
    for (let k = 1; k <= 10; k++) { await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0 + (x1 - x0) * k / 10, y: y0 + (y1 - y0) * k / 10, button: 'left', buttons: 1 }); await sleep(16); }
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: x1, y: y1, button: 'left', clickCount: 1 });
  }
  if (s.tap || s.tdrag) {   // {tap: [x, y]} 손가락 탭, {tdrag: [x0, y0, x1, y1]} 손가락으로 끌기
    const [x0, y0, x1 = x0, y1 = y0] = s.tap || s.tdrag, n = s.tap ? 0 : 10;
    await send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: x0, y: y0 }] });
    for (let k = 1; k <= n; k++) { await send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: x0 + (x1 - x0) * k / n, y: y0 + (y1 - y0) * k / n }] }); await sleep(16); }
    await send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  }
  if (s.wheel) {   // {wheel: [x, y, dx, dy, ctrl]} 휠·트랙패드 핀치(ctrl)
    const [x, y, dx, dy, ctrl] = s.wheel;
    await send('Input.dispatchMouseEvent', { type: 'mouseWheel', x, y, deltaX: dx, deltaY: dy, modifiers: ctrl ? 2 : 0 });
  }
  if (s.code) {   // {code: "Equal", key: "="} 물리 키
    await send('Input.dispatchKeyEvent', { type: 'keyDown', code: s.code, key: s.keyName || s.code });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', code: s.code, key: s.keyName || s.code });
  }
  if (s.navigate) { await send('Page.navigate', { url: s.navigate }); }
  if (s.wait) await sleep(s.wait);
  if (s.shot) { const r = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(join(outdir, s.shot + '.png'), Buffer.from(r.result.data, 'base64')); }
  if (s.clickSel) {
    const r = await send('Runtime.evaluate', { expression: `(()=>{const r=document.querySelector(${JSON.stringify(s.clickSel)}).getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()`, returnByValue: true });
    s.click = r.result.result.value;
  }
  if (s.burst) {   // 연속 촬영: {burst: "이름", n: 장수}
    const t0 = Date.now();
    for (let k = 0; k < (s.n || 10); k++) {
      const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 70 });
      writeFileSync(join(outdir, `${s.burst}_${String(k).padStart(2, '0')}_${Date.now() - t0}ms.jpg`), Buffer.from(r.result.data, 'base64'));
    }
  }
  for (const kind of ['moveTile', 'clickTile']) if (s[kind] !== undefined) {
    const r = await send('Runtime.evaluate', { expression: `(()=>{const r=G.rect(${s[kind]},false);return [r.left+r.width/2,r.top+r.height/2]})()`, returnByValue: true });
    s[kind === 'moveTile' ? 'move' : 'click'] = r.result.result.value;
    console.log(kind, s[kind], '->', r.result.result.value);
  }
  if (s.move) await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: s.move[0], y: s.move[1] });
  if (s.click) {
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: s.click[0], y: s.click[1] });
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: s.click[0], y: s.click[1], button: 'left', clickCount: 1 });
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: s.click[0], y: s.click[1], button: 'left', clickCount: 1 });
  }
  if (s.key) {
    const code = { ArrowUp: 38, ArrowDown: 40, ArrowLeft: 37, ArrowRight: 39, Escape: 27, ' ': 32 }[s.key] || s.key.charCodeAt(0);
    const kc = s.code || ({ ArrowUp: 'ArrowUp', ArrowDown: 'ArrowDown', ArrowLeft: 'ArrowLeft', ArrowRight: 'ArrowRight', Escape: 'Escape', ' ': 'Space', Enter: 'Enter', Backspace: 'Backspace', '/': 'Slash' }[s.key]
      || (/^[0-9]$/.test(s.key) ? 'Digit' + s.key : /^[a-z]$/i.test(s.key) ? 'Key' + s.key.toUpperCase() : undefined));
    const mods = s.shift ? 8 : 0;
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: s.key, code: kc, modifiers: mods, windowsVirtualKeyCode: code, text: s.key.length === 1 ? s.key : undefined });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: s.key, code: kc, modifiers: mods, windowsVirtualKeyCode: code });
  }
  if (s.type) await send('Input.insertText', { text: s.type });
  if (s.eval) { const r = await send('Runtime.evaluate', { expression: s.eval, returnByValue: true }); console.log('eval:', JSON.stringify(r.result.result?.value)); }
}
console.log(logs.length ? logs.join('\n') : 'no errors');
ws.close(); chrome.kill();
process.exit(0);
