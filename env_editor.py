"""Interactive GUI editor for creating and editing GridWorld environment configurations.

Usage:
    python env_editor.py
    python env_editor.py --config configs/env/default.json

Controls:
    - Left-click / drag : paint the selected cell type
    - Right-click / drag: erase (set cell to EMPTY)
    - Ctrl+N : new grid        Ctrl+O : open file
    - Ctrl+S : save            Ctrl+Shift+S : save as

Notes:
    Multiple start tiles and multiple goal tiles are both supported.
    On episode start the environment picks one start position at random.
    Reaching *any* goal tile terminates the episode.
    The Save command validates the configuration before writing to disk.
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Optional

from environments.objects import CellType

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

CELL_SIZE: int = 32       # pixels per grid cell
CANVAS_BG: str = "#888888"  # canvas background colour — visible as grid lines

# Hex colours matching the Pygame renderer
CELL_COLORS: dict[CellType, str] = {
    CellType.EMPTY:    "#F5F5F5",
    CellType.START:    "#5078DC",
    CellType.GOAL:     "#50C878",
    CellType.AGENT:    "#F5A02D",   # runtime only, not paintable
    CellType.WALL:     "#282828",
    CellType.HAZARD:   "#DC5050",
    CellType.SLIPPERY: "#5AD2F0",
}

CELL_LABELS: dict[CellType, str] = {
    CellType.EMPTY:    "Empty",
    CellType.START:    "Start",
    CellType.GOAL:     "Goal",
    CellType.AGENT:    "Agent",
    CellType.WALL:     "Wall",
    CellType.HAZARD:   "Hazard",
    CellType.SLIPPERY: "Slippery",
}

# Legible text colour on each cell background
CELL_FG: dict[CellType, str] = {
    CellType.EMPTY:    "#333333",
    CellType.START:    "#FFFFFF",
    CellType.GOAL:     "#FFFFFF",
    CellType.AGENT:    "#FFFFFF",
    CellType.WALL:     "#FFFFFF",
    CellType.HAZARD:   "#FFFFFF",
    CellType.SLIPPERY: "#1A3344",
}

# Cell types the user can paint (AGENT is a runtime overlay, not editable)
PAINTABLE: tuple[CellType, ...] = (
    CellType.EMPTY,
    CellType.START,
    CellType.GOAL,
    CellType.WALL,
    CellType.HAZARD,
    CellType.SLIPPERY,
)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class EditorApp(tk.Tk):
    """Full-featured GridWorld environment editor window."""

    def __init__(self, initial_config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.title("GridWorld Environment Editor")
        self.minsize(860, 540)

        # ---- Grid state ----
        self.grid_w: int = 10
        self.grid_h: int = 8
        self._grid: list[list[CellType]] = []

        # ---- Interaction state ----
        self._active_tool: CellType = CellType.WALL
        self._painting: bool = False
        self._erasing: bool = False

        # Canvas item IDs: _cell_ids[row][col] → int
        self._cell_ids: list[list[int]] = []

        # Current save path and dirty flag
        self._save_path: Optional[Path] = None
        self._dirty: bool = False

        # ---- Config parameters (StringVars for two-way Entry binding) ----
        self._var_grid_w     = tk.StringVar(value="10")
        self._var_grid_h     = tk.StringVar(value="8")
        self._var_slip_prob  = tk.StringVar(value="0.35")
        self._var_max_steps  = tk.StringVar(value="200")
        self._var_step_pen   = tk.StringVar(value="-0.01")
        self._var_goal_rew   = tk.StringVar(value="1.0")
        self._var_hazard_pen = tk.StringVar(value="-1.0")

        # _status_var must exist before _build_ui() because _setup_tools_panel
        # immediately calls _select_tool → _update_status during construction.
        self._status_var = tk.StringVar(value="Ready")

        # ---- Build UI then initialise with a blank grid ----
        self._build_ui()
        self._new_grid(self.grid_w, self.grid_h)

        if initial_config_path is not None:
            self._load_from_path(initial_config_path)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._setup_menu()
        # Status / notification bars must be packed before the main content frame
        # so tkinter reserves their space at the bottom first.
        self._setup_status_bar()

        # Three-column layout: tools | canvas | config
        outer = ttk.Frame(self, padding=4)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer, width=138)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        mid = ttk.Frame(outer)
        mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(outer, width=190)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        right.pack_propagate(False)

        self._setup_tools_panel(left)
        self._setup_canvas(mid)
        self._setup_config_panel(right)

    def _setup_menu(self) -> None:
        mb = tk.Menu(self)

        fm = tk.Menu(mb, tearoff=False)
        fm.add_command(label="New",        accelerator="Ctrl+N",       command=self._cmd_new)
        fm.add_command(label="Open…",      accelerator="Ctrl+O",       command=self._cmd_open)
        fm.add_separator()
        fm.add_command(label="Save",       accelerator="Ctrl+S",       command=self._cmd_save)
        fm.add_command(label="Save As…",   accelerator="Ctrl+Shift+S", command=self._cmd_save_as)
        fm.add_separator()
        fm.add_command(label="Quit", command=self._on_close)
        mb.add_cascade(label="File", menu=fm)

        em = tk.Menu(mb, tearoff=False)
        em.add_command(label="Clear Grid", command=self._cmd_clear_grid)
        em.add_separator()
        em.add_command(label="Validate",   command=self._cmd_validate)
        mb.add_cascade(label="Edit", menu=em)

        self.config(menu=mb)

        self.bind_all("<Control-n>", lambda _e: self._cmd_new())
        self.bind_all("<Control-o>", lambda _e: self._cmd_open())
        self.bind_all("<Control-s>", lambda _e: self._cmd_save())
        self.bind_all("<Control-S>", lambda _e: self._cmd_save_as())

    def _setup_tools_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="PAINT TOOL", font=("TkDefaultFont", 9, "bold")).pack(pady=(8, 4))

        self._tool_buttons: dict[CellType, tk.Button] = {}
        for ct in PAINTABLE:
            btn = tk.Button(
                parent,
                text=CELL_LABELS[ct],
                bg=CELL_COLORS[ct],
                fg=CELL_FG[ct],
                activebackground=CELL_COLORS[ct],
                activeforeground=CELL_FG[ct],
                relief=tk.FLAT,
                bd=2,
                width=13,
                cursor="hand2",
                command=lambda t=ct: self._select_tool(t),
            )
            btn.pack(pady=2, padx=6, fill=tk.X)
            self._tool_buttons[ct] = btn

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 6), padx=4)

        ttk.Label(parent, text="GRID SIZE", font=("TkDefaultFont", 9, "bold")).pack(pady=(0, 4))
        sf = ttk.Frame(parent)
        sf.pack(fill=tk.X, padx=6)

        ttk.Label(sf, text="W:").grid(row=0, column=0, sticky=tk.W)
        tk.Spinbox(sf, from_=3, to=80, width=5, textvariable=self._var_grid_w).grid(row=0, column=1)
        ttk.Label(sf, text="H:").grid(row=1, column=0, sticky=tk.W, pady=2)
        tk.Spinbox(sf, from_=3, to=60, width=5, textvariable=self._var_grid_h).grid(row=1, column=1)

        ttk.Button(parent, text="Resize Grid", command=self._cmd_resize).pack(
            pady=(8, 2), padx=6, fill=tk.X,
        )

        self._select_tool(CellType.WALL)

    def _setup_canvas(self, parent: ttk.Frame) -> None:
        """Central scrollable canvas for the grid."""
        self._canvas = tk.Canvas(
            parent,
            bg=CANVAS_BG,
            cursor="crosshair",
            highlightthickness=0,
        )
        h_sb = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self._canvas.xview)
        v_sb = ttk.Scrollbar(parent, orient=tk.VERTICAL,   command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=h_sb.set, yscrollcommand=v_sb.set)

        h_sb.pack(side=tk.BOTTOM, fill=tk.X)
        v_sb.pack(side=tk.RIGHT,  fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Paint with left button, erase with right button
        self._canvas.bind("<ButtonPress-1>",   self._on_lmb_press)
        self._canvas.bind("<B1-Motion>",       self._on_lmb_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self._canvas.bind("<ButtonPress-3>",   self._on_rmb_press)
        self._canvas.bind("<B3-Motion>",       self._on_rmb_drag)
        self._canvas.bind("<ButtonRelease-3>", self._on_rmb_release)
        self._canvas.bind("<Motion>",          self._on_mouse_move)

    def _setup_config_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="CONFIG", font=("TkDefaultFont", 9, "bold")).pack(pady=(8, 4))

        pf = ttk.LabelFrame(parent, text="Parameters", padding=6)
        pf.pack(fill=tk.X, padx=4, pady=2)

        param_rows: list[tuple[str, tk.StringVar]] = [
            ("Slip Prob:",    self._var_slip_prob),
            ("Max Steps:",    self._var_max_steps),
            ("Step Penalty:", self._var_step_pen),
            ("Goal Reward:",  self._var_goal_rew),
            ("Hazard Pen.:",  self._var_hazard_pen),
        ]
        for idx, (lbl, var) in enumerate(param_rows):
            ttk.Label(pf, text=lbl).grid(row=idx, column=0, sticky=tk.W, pady=1)
            ttk.Entry(pf, textvariable=var, width=9).grid(row=idx, column=1, padx=(4, 0), pady=1)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 6), padx=4)

        # Colour legend
        lf = ttk.LabelFrame(parent, text="Legend", padding=6)
        lf.pack(fill=tk.X, padx=4, pady=2)
        for ct in PAINTABLE:
            row_f = ttk.Frame(lf)
            row_f.pack(fill=tk.X, pady=1)
            tk.Label(row_f, bg=CELL_COLORS[ct], width=3, relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Label(row_f, text=CELL_LABELS[ct]).pack(side=tk.LEFT)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 6), padx=4)

        bf = ttk.Frame(parent)
        bf.pack(fill=tk.X, padx=4)
        ttk.Button(bf, text="Validate", command=self._cmd_validate).pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="Save",     command=self._cmd_save).pack(fill=tk.X, pady=2)

    def _setup_status_bar(self) -> None:
        bottom = tk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        bottom.columnconfigure(0, weight=1)

        # Notification bar: hidden by default, shown via _notify().
        self._notif_label = tk.Label(
            bottom,
            text="",
            anchor=tk.W,
            padx=8,
            pady=3,
            font=("TkDefaultFont", 9),
        )

        ttk.Label(
            bottom,
            textvariable=self._status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(6, 2),
        ).grid(row=1, column=0, sticky=tk.EW)

    # -----------------------------------------------------------------------
    # Grid management
    # -----------------------------------------------------------------------

    def _new_grid(self, width: int, height: int) -> None:
        """Replace the current grid with a blank one of the given dimensions."""
        self.grid_w = width
        self.grid_h = height
        self._grid = [[CellType.EMPTY] * width for _ in range(height)]
        self._full_redraw()

    def _full_redraw(self) -> None:
        """Delete all canvas items and repaint every cell from scratch."""
        self._canvas.delete("all")
        self._canvas.configure(
            scrollregion=(0, 0, self.grid_w * CELL_SIZE, self.grid_h * CELL_SIZE)
        )
        self._cell_ids = [[0] * self.grid_w for _ in range(self.grid_h)]
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                item = self._canvas.create_rectangle(
                    x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE,
                    fill=CELL_COLORS[self._grid[r][c]],
                    outline=CANVAS_BG,
                    width=1,
                )
                self._cell_ids[r][c] = item

    def _redraw_cell(self, row: int, col: int) -> None:
        """Fast single-cell colour update without recreating the canvas item."""
        self._canvas.itemconfig(
            self._cell_ids[row][col],
            fill=CELL_COLORS[self._grid[row][col]],
        )

    def _canvas_to_grid(self, ex: int, ey: int) -> Optional[tuple[int, int]]:
        """Convert a screen-event coordinate to ``(row, col)``, accounting for scroll."""
        cx = int(self._canvas.canvasx(ex))
        cy = int(self._canvas.canvasy(ey))
        col = cx // CELL_SIZE
        row = cy // CELL_SIZE
        if 0 <= row < self.grid_h and 0 <= col < self.grid_w:
            return row, col
        return None

    def _paint_at(self, ex: int, ey: int, ct: CellType) -> None:
        pos = self._canvas_to_grid(ex, ey)
        if pos is None:
            return
        r, c = pos
        if self._grid[r][c] != ct:
            self._grid[r][c] = ct
            self._redraw_cell(r, c)
            self._mark_dirty()

    def _select_tool(self, ct: CellType) -> None:
        self._active_tool = ct
        for t, btn in self._tool_buttons.items():
            btn.config(relief=tk.RAISED if t == ct else tk.FLAT,
                       bd=3 if t == ct else 2)
        self._update_status()

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            name = self._save_path.name if self._save_path else "Untitled"
            self.title(f"GridWorld Environment Editor — {name} *")

    # -----------------------------------------------------------------------
    # Mouse handlers
    # -----------------------------------------------------------------------

    def _on_lmb_press(self, e: tk.Event) -> None:       # type: ignore[type-arg]
        self._painting = True
        self._paint_at(e.x, e.y, self._active_tool)

    def _on_lmb_drag(self, e: tk.Event) -> None:        # type: ignore[type-arg]
        if self._painting:
            self._paint_at(e.x, e.y, self._active_tool)

    def _on_lmb_release(self, e: tk.Event) -> None:     # type: ignore[type-arg]
        self._painting = False

    def _on_rmb_press(self, e: tk.Event) -> None:       # type: ignore[type-arg]
        self._erasing = True
        self._paint_at(e.x, e.y, CellType.EMPTY)

    def _on_rmb_drag(self, e: tk.Event) -> None:        # type: ignore[type-arg]
        if self._erasing:
            self._paint_at(e.x, e.y, CellType.EMPTY)

    def _on_rmb_release(self, e: tk.Event) -> None:     # type: ignore[type-arg]
        self._erasing = False

    def _on_mouse_move(self, e: tk.Event) -> None:      # type: ignore[type-arg]
        self._update_status(self._canvas_to_grid(e.x, e.y))

    def _update_status(self, pos: Optional[tuple[int, int]] = None) -> None:
        tool = CELL_LABELS[self._active_tool]
        dims = f"{self.grid_w}×{self.grid_h}"
        if pos is not None:
            r, c = pos
            cell_name = CELL_LABELS.get(self._grid[r][c], self._grid[r][c].name)
            self._status_var.set(f"({r}, {c})  Cell: {cell_name}  |  Tool: {tool}  |  {dims}")
        else:
            self._status_var.set(f"Tool: {tool}  |  {dims}")

    def _notify(self, message: str, level: str = "info") -> None:
        """Show an inline notification above the status bar.

        level: ``'ok'`` | ``'error'`` | ``'warn'`` | ``'info'``
        """
        styles: dict[str, tuple[str, str]] = {
            "ok":    ("#155724", "#D4EDDA"),
            "error": ("#721C24", "#F8D7DA"),
            "warn":  ("#856404", "#FFF3CD"),
            "info":  ("#0C5460", "#D1ECF1"),
        }
        fg, bg = styles.get(level, styles["info"])
        self._notif_label.configure(text=f"  {message}", foreground=fg, background=bg)
        self._notif_label.grid(row=0, column=0, sticky=tk.EW)
        if level != "error":
            self.after(5000, self._dismiss_notif)

    def _dismiss_notif(self) -> None:
        """Hide the notification bar."""
        self._notif_label.grid_remove()

    # -----------------------------------------------------------------------
    # Validation and serialisation
    # -----------------------------------------------------------------------

    def _validate(self) -> list[str]:
        """Return a list of error strings; empty list means the config is valid."""
        errors: list[str] = []

        starts = [
            (r, c) for r in range(self.grid_h) for c in range(self.grid_w)
            if self._grid[r][c] == CellType.START
        ]
        goals = [
            (r, c) for r in range(self.grid_h) for c in range(self.grid_w)
            if self._grid[r][c] == CellType.GOAL
        ]

        if not starts:
            errors.append("No start tile (blue) placed — add at least one.")
        if not goals:
            errors.append("No goal tile (green) placed — add at least one.")

        num_checks: list[tuple[str, tk.StringVar, bool]] = [
            ("Slip probability",  self._var_slip_prob,  False),
            ("Max steps",         self._var_max_steps,  True),
            ("Step penalty",      self._var_step_pen,   False),
            ("Goal reward",       self._var_goal_rew,   False),
            ("Hazard penalty",    self._var_hazard_pen, False),
        ]
        for name, var, must_be_int in num_checks:
            raw = var.get().strip()
            try:
                val = float(raw)
            except ValueError:
                errors.append(f"{name} must be a number (got '{raw}').")
                continue
            if name == "Slip probability" and not 0.0 <= val <= 1.0:
                errors.append("Slip probability must be between 0.0 and 1.0.")
            if must_be_int and (val != int(val) or int(val) < 1):
                errors.append(f"{name} must be a positive integer.")

        return errors

    def _to_dict(self) -> dict:
        """Serialise the current editor state to a plain dict (JSON-ready)."""
        starts, goals, walls, hazards, slippery = [], [], [], [], []
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                ct = self._grid[r][c]
                if   ct == CellType.START:    starts.append([r, c])
                elif ct == CellType.GOAL:     goals.append([r, c])
                elif ct == CellType.WALL:     walls.append([r, c])
                elif ct == CellType.HAZARD:   hazards.append([r, c])
                elif ct == CellType.SLIPPERY: slippery.append([r, c])
        return {
            "width":            self.grid_w,
            "height":           self.grid_h,
            "start_positions":  starts,
            "goal_positions":   goals,
            "walls":            walls,
            "hazards":          hazards,
            "slippery_tiles":   slippery,
            "slip_probability": float(self._var_slip_prob.get()),
            "max_steps":        int(float(self._var_max_steps.get())),
            "step_penalty":     float(self._var_step_pen.get()),
            "goal_reward":      float(self._var_goal_rew.get()),
            "hazard_penalty":   float(self._var_hazard_pen.get()),
        }

    def _from_dict(self, data: dict) -> None:
        """Populate the editor from a config dict (new *and* legacy format)."""
        w = int(data["width"])
        h = int(data["height"])
        self.grid_w, self.grid_h = w, h
        self._var_grid_w.set(str(w))
        self._var_grid_h.set(str(h))
        self._grid = [[CellType.EMPTY] * w for _ in range(h)]

        def _set(r: int, c: int, ct: CellType) -> None:
            if 0 <= r < h and 0 <= c < w:
                self._grid[r][c] = ct

        # Support legacy single-position keys as well as the new list keys.
        if "start_positions" in data:
            for r, c in data["start_positions"]:
                _set(r, c, CellType.START)
        elif "start_pos" in data:
            r, c = data["start_pos"]
            _set(r, c, CellType.START)

        if "goal_positions" in data:
            for r, c in data["goal_positions"]:
                _set(r, c, CellType.GOAL)
        elif "goal_pos" in data:
            r, c = data["goal_pos"]
            _set(r, c, CellType.GOAL)

        for r, c in data.get("walls", []):
            _set(r, c, CellType.WALL)
        for r, c in data.get("hazards", []):
            _set(r, c, CellType.HAZARD)
        for r, c in data.get("slippery_tiles", []):
            _set(r, c, CellType.SLIPPERY)

        self._var_slip_prob.set(str(data.get("slip_probability", 0.35)))
        self._var_max_steps.set(str(data.get("max_steps", 200)))
        self._var_step_pen.set(str(data.get("step_penalty", -0.01)))
        self._var_goal_rew.set(str(data.get("goal_reward", 1.0)))
        self._var_hazard_pen.set(str(data.get("hazard_penalty", -1.0)))

        self._full_redraw()

    def _load_from_path(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._from_dict(data)
            self._save_path = path
            self._dirty = False
            self.title(f"GridWorld Environment Editor — {path.name}")
            self._status_var.set(f"Loaded: {path}")
            self._notify(f"Loaded {path.name}", "ok")
        except Exception as exc:
            self._notify(f"Could not open file: {exc}", "error")

    # -----------------------------------------------------------------------
    # Command handlers
    # -----------------------------------------------------------------------

    def _cmd_new(self) -> None:
        if not self._ask_discard():
            return
        try:
            w = max(3, int(self._var_grid_w.get()))
            h = max(3, int(self._var_grid_h.get()))
        except ValueError:
            w, h = 10, 8
        self._new_grid(w, h)
        self._save_path = None
        self._dirty = False
        self.title("GridWorld Environment Editor — Untitled")
        self._status_var.set(f"New {w}×{h} grid created.")

    def _cmd_open(self) -> None:
        if not self._ask_discard():
            return
        path_str = filedialog.askopenfilename(
            title="Open environment configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="configs/env",
        )
        if path_str:
            self._load_from_path(Path(path_str))

    def _cmd_save(self) -> None:
        if self._save_path is None:
            self._cmd_save_as()
        else:
            self._save_to_path(self._save_path)

    def _cmd_save_as(self) -> None:
        path_str = filedialog.asksaveasfilename(
            title="Save environment configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="configs/env",
        )
        if path_str:
            self._save_to_path(Path(path_str))

    def _save_to_path(self, path: Path) -> None:
        """Validate then write the config.  Aborts with a dialog on any error."""
        errors = self._validate()
        if errors:
            first = errors[0]
            extra = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
            self._notify(f"Cannot save — {first}{extra}", "error")
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._to_dict(), indent=2), encoding="utf-8")
            self._save_path = path
            self._dirty = False
            self.title(f"GridWorld Environment Editor — {path.name}")
            self._status_var.set(f"Saved: {path}")
            self._notify(f"Saved to {path.name}", "ok")
        except Exception as exc:
            self._notify(f"Save failed: {exc}", "error")

    def _cmd_validate(self) -> None:
        errors = self._validate()
        if errors:
            first = errors[0]
            extra = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
            self._notify(f"{first}{extra}", "error")
        else:
            n_starts = sum(
                1 for r in range(self.grid_h) for c in range(self.grid_w)
                if self._grid[r][c] == CellType.START
            )
            n_goals = sum(
                1 for r in range(self.grid_h) for c in range(self.grid_w)
                if self._grid[r][c] == CellType.GOAL
            )
            self._notify(
                f"Valid — {n_starts} start tile(s), {n_goals} goal tile(s)",
                "ok",
            )

    def _cmd_clear_grid(self) -> None:
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                self._grid[r][c] = CellType.EMPTY
        self._full_redraw()
        self._mark_dirty()
        self._notify("Grid cleared.", "info")

    def _cmd_resize(self) -> None:
        """Resize the grid, preserving existing content in the overlapping region."""
        try:
            nw = int(self._var_grid_w.get())
            nh = int(self._var_grid_h.get())
            if nw < 3 or nh < 3:
                raise ValueError("Minimum grid size is 3×3.")
        except ValueError as exc:
            self._notify(str(exc), "error")
            return

        old, ow, oh = self._grid, self.grid_w, self.grid_h
        self.grid_w, self.grid_h = nw, nh
        self._grid = [[CellType.EMPTY] * nw for _ in range(nh)]
        for r in range(min(oh, nh)):
            for c in range(min(ow, nw)):
                self._grid[r][c] = old[r][c]
        self._full_redraw()
        self._mark_dirty()
        self._status_var.set(f"Grid resized to {nw}×{nh}.")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _ask_discard(self) -> bool:
        """Always permit discarding; the title-bar asterisk warns of unsaved changes."""
        return True

    def _on_close(self) -> None:
        if self._ask_discard():
            self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GridWorld environment editor.")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON configuration file to open on startup.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = EditorApp(initial_config_path=args.config)
    app.mainloop()


if __name__ == "__main__":
    main()
