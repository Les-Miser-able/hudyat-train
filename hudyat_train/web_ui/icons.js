'use strict';

    const icons={
      mark:'<path d="M12 3v6m0 6v6M3 12h6m6 0h6M5.6 5.6l4.2 4.2m4.4 4.4 4.2 4.2m0-12.8-4.2 4.2m-4.4 4.4-4.2 4.2"/><path d="m12 7 5 5-5 5-5-5Z"/>',
      home:'<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1Z"/>',
      dataset:'<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
      extract:'<path d="M5 3h10l4 4v7M14 3v5h5M5 3v18h7m5-8v8m-3-3 3 3 3-3"/><path d="M8 11h3m-3 4h3"/>',
      train:'<path d="m12 3 9 5-9 5-9-5Zm-9 9 9 5 9-5m-18 5 9 5 9-5"/>',
      test:'<rect x="3" y="5" width="14" height="14" rx="3"/><path d="m17 10 4-3v10l-4-3m-8-4 4 2-4 2Z"/>',
      results:'<path d="M4 3v17h17M8 15v-4m5 4V6m5 9v-7"/>',
      export:'<path d="M12 3v12m-4-4 4 4 4-4M5 14v6h14v-6"/>',
      settings:'<path d="m10 3-.7 2.5-2 .9L5 5.7 3 9l1.8 1.8v2.4L3 15l2 3.3 2.3-.7 2 .9L10 21h4l.7-2.5 2-.9 2.3.7 2-3.3-1.8-1.8v-2.4L21 9l-2-3.3-2.3.7-2-.9L14 3Z"/><circle cx="12" cy="12" r="3"/>',
      arrow:'<path d="M5 12h14m-5-5 5 5-5 5"/>', chevron:'<path d="m9 5 7 7-7 7"/>', left:'<path d="m15 5-7 7 7 7"/>', down:'<path d="m7 10 5 5 5-5"/>', plus:'<path d="M12 5v14M5 12h14"/>',
      play:'<path d="m9 5 11 7-11 7Z"/>', pause:'<path d="M8 5v14M16 5v14"/>', stop:'<rect x="6" y="6" width="12" height="12" rx="1"/>', bell:'<path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9m-11 12a2 2 0 0 0 4 0"/>',
      folder:'<path d="M3 7V5h6l2 2h10v13H3Zm0 4h18"/>', check:'<path d="m5 12 4 4 10-10"/>', clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>', signers:'<circle cx="9" cy="8" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3m1-16a3 3 0 0 1 0 6m2 4a5 5 0 0 1 3 4v2"/>',
      sign:'<path d="M8 13V6a1.5 1.5 0 0 1 3 0v5-7a1.5 1.5 0 0 1 3 0v7-6a1.5 1.5 0 0 1 3 0v7-4a1.5 1.5 0 0 1 3 0v8c0 4-3 6-6 6h-2c-2 0-3-1-4-3l-4-5a1.5 1.5 0 0 1 2-2Z"/>',
      search:'<circle cx="10" cy="10" r="6.5"/><path d="m15 15 5 5"/>', refresh:'<path d="M20 7v5h-5M4 17v-5h5m-4-5a8 8 0 0 1 14-1l1 6M4 12l1 6a8 8 0 0 0 14-1"/>', close:'<path d="m6 6 12 12M6 18 18 6"/>', shield:'<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Z"/><path d="m8 12 3 3 5-6"/>', code:'<path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18"/>'
    };
    const icon=(name,cls='')=>`<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true">${icons[name]||icons.train}</svg>`;
    const brand=()=>`<div class="brand" aria-label="HUDYAT">${icon('mark')}</div>`;
    const hand=()=>`<svg class="hand" viewBox="0 0 200 240" fill="none" aria-hidden="true"><g stroke="currentColor" stroke-width="1.2"><path d="m87 219-39-69-25-25 10-15 35 23-4-78 13-4 13 70-1-94 14-2 11 94 8-106 14 1-2 106 18-87 14 5-15 107 13-51 13 7-12 85-26 42Z" fill="currentColor" fill-opacity=".035"/><path d="m87 219 7-64-26-22 22-12 24-2 20 2 17 20 8 31m-65-17 20-36 20 2-12 57-35 41m7-64 28 23 29-37m-57 14-46-5m46 5-13-34m33-2-10-94m30 96 2-106m15 126 15-102m-72 116 13-3 17 2 15 9 10 11m-47-50 13-1 13-1 16 3 17 8m-65-59 12-2m14-20 14-1m16-3 13 1m16 19 12 3m-49 136 19 31"/></g><g fill="currentColor">${[[87,219],[48,150],[23,125],[33,110],[68,133],[64,55],[77,51],[90,121],[89,27],[103,25],[114,119],[122,13],[136,14],[134,121],[152,34],[166,39],[151,141],[164,90],[177,97],[165,182],[122,178],[94,155],[142,222],[159,172],[81,78],[109,63],[135,59],[159,83],[170,130]].map(([x,y])=>`<circle cx="${x}" cy="${y}" r="2.5"/>`).join('')}</g></svg>`;
