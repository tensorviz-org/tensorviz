// Sync colour scheme with the parent docs site (same origin).
(function(){
  function apply(){try{document.documentElement.classList.toggle('dark',window.parent.document.documentElement.classList.contains('dark'))}catch(e){document.documentElement.classList.toggle('dark',matchMedia('(prefers-color-scheme: dark)').matches)}}
  apply();
  try{new MutationObserver(apply).observe(window.parent.document.documentElement,{attributes:true,attributeFilter:['class']})}catch(e){}
})();
