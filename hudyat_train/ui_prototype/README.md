# HUDYAT Train — UI prototype

Question: how can the trainer's existing workflow use the reference's warm, dark dashboard, orange focal panels, and mint status accents?

This is a throwaway, offline design preview next to the native desktop UI. There is no existing browser trainer page to host it. All numbers are synthetic, all actions are simulated in memory, and no datasets, cameras, models, or settings are accessed.

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory hudyat_train/ui_prototype
```

Open **http://127.0.0.1:8765/**. You can also double-click `index.html`.

- `?variant=A`: **Studio** — chart-led dashboard, closest to the reference.
- `?variant=B`: **Focus** — guided training setup with a persistent run summary.
- `?variant=C`: **Runroom** — dense experiment comparison and model inspection.

Use the floating arrows or keyboard Left/Right to switch. The current variant is stored in the URL; other state is temporary. **Demo state** exposes the complete state. The switcher is only shown on a local/file preview.

Try the sidebar, category and input selectors, chart tabs, training configuration, simulated run/pause/cancel, dataset scanning/extraction, run inspection, mock sign testing, and export preview. Export previews do not write files.

The working application now follows **A / Studio** in the browser, using this prototype's HTML/CSS visual language with real Python workers and saved reports. The user selected browser mode after the native adaptation did not match the prototype closely enough. The original options are preserved in commit `5974bab` (`PROTOTYPE`); this folder remains a throwaway reference. Production assets live in `hudyat_train/web_ui/` and contain no variant switcher or simulated actions.
