/* Keep module-loading failures visible even when app.js cannot start. */
(() => {
  const status = document.getElementById('startup-status');
  function failed(error) {
    status.hidden = false;
    status.setAttribute('role', 'alert');
    status.textContent = '화면을 불러오지 못했습니다: ' + (error?.message || String(error)) + ' · 새로고침 후에도 같으면 이 문구를 알려주세요.';
  }
  window.addEventListener('error', event => failed(event.error || event.message));
  window.addEventListener('unhandledrejection', event => failed(event.reason));
  status.textContent = '화면 프로그램을 불러오는 중…';
  async function load() {
    for (const name of ['navigation.mjs','chat.mjs','mask.mjs','pipeline.mjs','app.js']) {
      try {
        const module = await import('./' + name + '?v=20261107-ui32');
        if (name === 'app.js') await module.ready;
      } catch(error) { throw new Error(name + ': ' + error.message); }
    }
    status.hidden = true;
  }
  load().catch(failed);
})();
