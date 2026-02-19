"""Interactive GUI editor for creating and editing GridWorld environment configurations.

Usage:
    python env_editor.py
    python env_editor.py --config configs/env/default.json

Controls:
    - Left-click / drag : paint the selected cell type
    - Right-click / drag: erase (set cell to EMPTY)
    - Ctrl+N : new grid        Ctrl+O : load file
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

# Tkinter's Event type is generic in typeshed, so provide a concrete alias
# for bound callback signatures.
TkMouseEvent = tk.Event[tk.Misc]


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class EditorApp(tk.Tk):
    """Full-featured GridWorld environment editor window."""

    def __init__(self, initial_config_path: Optional[Path] = None) -> None:
        """Initialize the editor window, state containers, and UI widgets.

        Args:
            initial_config_path (Optional[Path]): Optional config file to load
                immediately after creating the default blank grid.

        Returns:
            None: This constructor initializes the application instance in place.
        """
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
        """Assemble the top-level window layout and child panels.

        Returns:
            None: Widgets are created and attached to the main window.
        """
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
        """Create menu entries and keyboard shortcuts for common file actions.

        Returns:
            None: Menu state is bound to this window instance.
        """
        mb = tk.Menu(self)

        fm = tk.Menu(mb, tearoff=False)
        fm.add_command(label="New",        accelerator="Ctrl+N",       command=self._cmd_new)
        fm.add_command(label="Load…",      accelerator="Ctrl+O",       command=self._cmd_load)
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
        self.bind_all("<Control-o>", lambda _e: self._cmd_load())
        self.bind_all("<Control-s>", lambda _e: self._cmd_save())
        self.bind_all("<Control-S>", lambda _e: self._cmd_save_as())

    def _setup_tools_panel(self, parent: ttk.Frame) -> None:
        """Create the left panel with paint tools and grid resize controls.

        Args:
            parent (ttk.Frame): Parent container where the tools panel is added.

        Returns:
            None: The panel widgets are created and packed into ``parent``.
        """
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
        """Create the central scrollable canvas used for grid drawing.

        Args:
            parent (ttk.Frame): Parent container where canvas and scrollbars are placed.

        Returns:
            None: Canvas widgets are created, packed, and event-bound.
        """
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
        """Create the right panel with file actions, parameters, and legend.

        Args:
            parent (ttk.Frame): Parent container where the config panel is added.

        Returns:
            None: The panel widgets are created and packed into ``parent``.
        """
        ttk.Label(parent, text="CONFIG", font=("TkDefaultFont", 9, "bold")).pack(pady=(8, 4))

        file_actions = ttk.LabelFrame(parent, text="File", padding=6)
        file_actions.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(file_actions, text="New", command=self._cmd_new).pack(fill=tk.X, pady=2)
        ttk.Button(file_actions, text="Load", command=self._cmd_load).pack(fill=tk.X, pady=2)
        ttk.Button(file_actions, text="Save As", command=self._cmd_save_as).pack(fill=tk.X, pady=2)

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
        """Create the bottom notification area and persistent status line.

        Returns:
            None: Bottom status widgets are initialized and attached.
        """
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
        """Replace the current grid with a blank grid of the requested size.

        Args:
            width (int): Number of columns in the new grid.
            height (int): Number of rows in the new grid.

        Returns:
            None: The internal grid state is reset and fully redrawn.
        """
        self.grid_w = width
        self.grid_h = height
        self._grid = [[CellType.EMPTY] * width for _ in range(height)]
        self._full_redraw()

    def _full_redraw(self) -> None:
        """Delete all canvas items and repaint every cell from scratch.

        Returns:
            None: Canvas item ids are recreated to match current grid state.
        """
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
        """Update one cell's fill color without recreating the canvas rectangle.

        Args:
            row (int): Grid row index of the cell to repaint.
            col (int): Grid column index of the cell to repaint.

        Returns:
            None: The existing canvas item is updated in place.
        """
        self._canvas.itemconfig(
            self._cell_ids[row][col],
            fill=CELL_COLORS[self._grid[row][col]],
        )

    def _canvas_to_grid(self, event_x: int, event_y: int) -> Optional[tuple[int, int]]:
        """Map canvas event coordinates to grid indices.

        Args:
            event_x (int): Mouse x coordinate relative to the canvas widget.
            event_y (int): Mouse y coordinate relative to the canvas widget.

        Returns:
            Optional[tuple[int, int]]: ``(row, col)`` for in-bounds coordinates,
            otherwise ``None``.
        """
        cx = int(self._canvas.canvasx(event_x))
        cy = int(self._canvas.canvasy(event_y))
        col = cx // CELL_SIZE
        row = cy // CELL_SIZE
        if 0 <= row < self.grid_h and 0 <= col < self.grid_w:
            return row, col
        return None

    def _paint_at(self, event_x: int, event_y: int, cell_type: CellType) -> None:
        """
        Apply `cell_type` to the cell under the mouse event coordinates.
        
        Args:
            event_x (int): Mouse event x coordinate relative to the canvas widget.
            event_y (int): Mouse event y coordinate relative to the canvas widget.
            cell_type (CellType): The cell type to paint at the hovered location.

        Returns:
            None: The target cell is updated only when its type changes.
        """
        pos = self._canvas_to_grid(event_x, event_y)
        if pos is None:
            return
        row, column = pos
        if self._grid[row][column] != cell_type:
            self._grid[row][column] = cell_type
            self._redraw_cell(row, column)
            self._mark_dirty()

    def _select_tool(self, cell_type: CellType) -> None:
        """
        Activate a paint tool and update button styles and status text.

        Args:
            cell_type (CellType): Paint tool to activate for subsequent edits.

        Returns:
            None: Internal tool state and button visuals are updated.
        """
        self._active_tool = cell_type
        for tool, button in self._tool_buttons.items():
            button.config(
                relief=tk.RAISED if tool == cell_type else tk.FLAT,
                bd=3 if tool == cell_type else 2,
            )
        self._update_status()

    def _mark_dirty(self) -> None:
        """Mark the editor as modified and show an asterisk in the title.

        Returns:
            None: Dirty state and window title are updated once per edit session.
        """
        if not self._dirty:
            self._dirty = True
            name = self._save_path.name if self._save_path else "Untitled"
            self.title(f"GridWorld Environment Editor — {name} *")

    # -----------------------------------------------------------------------
    # Mouse handlers
    # -----------------------------------------------------------------------

    def _on_lmb_press(self, event: TkMouseEvent) -> None:
        """Start paint mode and apply the active tool to the first hovered cell.

        Args:
            event (TkMouseEvent): Tkinter mouse event produced by left-button press.

        Returns:
            None: Paint mode is enabled and one paint operation is attempted.
        """
        self._painting = True
        self._paint_at(event.x, event.y, self._active_tool)

    def _on_lmb_drag(self, event: TkMouseEvent) -> None:
        """Continue painting while the left mouse button remains pressed.

        Args:
            event (TkMouseEvent): Tkinter mouse event produced by left-button drag.

        Returns:
            None: Paint operations continue while paint mode is active.
        """
        if self._painting:
            self._paint_at(event.x, event.y, self._active_tool)

    def _on_lmb_release(self, _event: TkMouseEvent) -> None:
        """End left-button paint mode.

        Args:
            _event (TkMouseEvent): Tkinter mouse event produced by left-button release.

        Returns:
            None: Paint mode is disabled.
        """
        self._painting = False

    def _on_rmb_press(self, event: TkMouseEvent) -> None:
        """Start erase mode and clear the first hovered cell to EMPTY.

        Args:
            event (TkMouseEvent): Tkinter mouse event produced by right-button press.

        Returns:
            None: Erase mode is enabled and one erase operation is attempted.
        """
        self._erasing = True
        self._paint_at(event.x, event.y, CellType.EMPTY)

    def _on_rmb_drag(self, event: TkMouseEvent) -> None:
        """Continue erasing while the right mouse button remains pressed.

        Args:
            event (TkMouseEvent): Tkinter mouse event produced by right-button drag.

        Returns:
            None: Erase operations continue while erase mode is active.
        """
        if self._erasing:
            self._paint_at(event.x, event.y, CellType.EMPTY)

    def _on_rmb_release(self, _event: TkMouseEvent) -> None:
        """End right-button erase mode.

        Args:
            _event (TkMouseEvent): Tkinter mouse event produced by right-button release.

        Returns:
            None: Erase mode is disabled.
        """
        self._erasing = False

    def _on_mouse_move(self, event: TkMouseEvent) -> None:
        """Refresh status text with the cell currently under the cursor.

        Args:
            event (TkMouseEvent): Tkinter mouse motion event.

        Returns:
            None: Status text is refreshed with cursor context.
        """
        self._update_status(self._canvas_to_grid(event.x, event.y))

    def _update_status(self, pos: Optional[tuple[int, int]] = None) -> None:
        """Render context status: cursor cell, active tool, and grid dimensions.

        Args:
            pos (Optional[tuple[int, int]]): Current ``(row, col)`` cursor position,
                or ``None`` when cursor position is unavailable.

        Returns:
            None: Status bar text is updated in-place.
        """
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

        Args:
            message (str): Notification text shown to the user.
            level (str): Visual style key. Supported values are ``'ok'``,
                ``'error'``, ``'warn'``, and ``'info'``.

        Returns:
            None: Notification bar is displayed and may auto-dismiss.
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
        """Hide the notification bar.

        Returns:
            None: Notification row is removed from the layout.
        """
        self._notif_label.grid_remove()

    # -----------------------------------------------------------------------
    # Validation and serialisation
    # -----------------------------------------------------------------------

    def _validate(self) -> list[str]:
        """Validate current grid and numeric parameters.

        Returns:
            list[str]: Validation errors. Empty list means the config is valid.
        """
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
        """Serialize the current editor state to a JSON-ready dictionary.

        Returns:
            dict: Environment config payload compatible with project config files.
        """
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
            "slip_probability": float(self._var_slip_prob.get()),
            "max_steps":        int(float(self._var_max_steps.get())),
            "step_penalty":     float(self._var_step_pen.get()),
            "goal_reward":      float(self._var_goal_rew.get()),
            "hazard_penalty":   float(self._var_hazard_pen.get()),
            "start_positions":  starts,
            "goal_positions":   goals,
            "walls":            walls,
            "hazards":          hazards,
            "slippery_tiles":   slippery,
        }

    @staticmethod
    def _format_tile_rows(tile_rows: list[list[int]], indent: str = "    ") -> list[str]:
        """Format tile coordinates as one row per tile: ``[r, c]``.

        Args:
            tile_rows (list[list[int]]): Tile coordinate pairs as ``[row, col]`` lists.
            indent (str): Prefix added to each formatted coordinate line.

        Returns:
            list[str]: Formatted coordinate lines, one entry per tile.
        """

        return [f"{indent}[{row}, {col}]" for row, col in tile_rows]

    def _format_config_json(self, payload: dict) -> str:
        """Format config JSON for readability.

        Rules:
            - Numeric settings first.
            - Tile lists use one line per tile coordinate.

        Args:
            payload (dict): Configuration dictionary produced by ``_to_dict``.

        Returns:
            str: Pretty-printed JSON text with stable key ordering.
        """

        numeric_keys: tuple[str, ...] = (
            "width",
            "height",
            "slip_probability",
            "max_steps",
            "step_penalty",
            "goal_reward",
            "hazard_penalty",
        )
        tile_keys: tuple[str, ...] = (
            "start_positions",
            "goal_positions",
            "walls",
            "hazards",
            "slippery_tiles",
        )

        lines: list[str] = ["{"]

        for key in numeric_keys:
            value = json.dumps(payload[key])
            lines.append(f'  "{key}": {value},')

        for index, key in enumerate(tile_keys):
            tile_rows = payload[key]
            is_last_block = index == len(tile_keys) - 1
            trailing_comma = '' if is_last_block else ','

            if not tile_rows:
                lines.append(f'  "{key}": []{trailing_comma}')
                continue

            lines.append(f'  "{key}": [')
            for row_index, row_line in enumerate(self._format_tile_rows(tile_rows)):
                is_last_row = row_index == len(tile_rows) - 1
                lines.append(f"{row_line}{'' if is_last_row else ','}")
            lines.append(f"  ]{trailing_comma}")

        lines.append("}")
        return "\n".join(lines) + "\n"

    def _from_dict(self, config_data: dict) -> None:
        """Populate editor state from a config dictionary.

        Supports both modern list-based keys and legacy single-position keys.

        Args:
            config_data (dict): Parsed environment configuration dictionary.

        Returns:
            None: Grid data and parameter fields are replaced with loaded values.
        """
        w = int(config_data["width"])
        h = int(config_data["height"])
        self.grid_w, self.grid_h = w, h
        self._var_grid_w.set(str(w))
        self._var_grid_h.set(str(h))
        self._grid = [[CellType.EMPTY] * w for _ in range(h)]

        def _set(row: int, col: int, cell_type: CellType) -> None:
            """Safely assign a cell type when coordinates are in bounds.

            Args:
                row (int): Target row index.
                col (int): Target column index.
                cell_type (CellType): Value to assign to the target cell.

            Returns:
                None: Assignment is performed only for in-bounds indices.
            """
            if 0 <= row < h and 0 <= col < w:
                self._grid[row][col] = cell_type

        # Support legacy single-position keys as well as the new list keys.
        if "start_positions" in config_data:
            for r, c in config_data["start_positions"]:
                _set(r, c, CellType.START)
        elif "start_pos" in config_data:
            r, c = config_data["start_pos"]
            _set(r, c, CellType.START)

        if "goal_positions" in config_data:
            for r, c in config_data["goal_positions"]:
                _set(r, c, CellType.GOAL)
        elif "goal_pos" in config_data:
            r, c = config_data["goal_pos"]
            _set(r, c, CellType.GOAL)

        for r, c in config_data.get("walls", []):
            _set(r, c, CellType.WALL)
        for r, c in config_data.get("hazards", []):
            _set(r, c, CellType.HAZARD)
        for r, c in config_data.get("slippery_tiles", []):
            _set(r, c, CellType.SLIPPERY)

        self._var_slip_prob.set(str(config_data.get("slip_probability", 0.35)))
        self._var_max_steps.set(str(config_data.get("max_steps", 200)))
        self._var_step_pen.set(str(config_data.get("step_penalty", -0.01)))
        self._var_goal_rew.set(str(config_data.get("goal_reward", 1.0)))
        self._var_hazard_pen.set(str(config_data.get("hazard_penalty", -1.0)))

        self._full_redraw()

    def _load_from_path(self, path: Path) -> None:
        """Load a JSON config file, update editor state, and report status.

        Args:
            path (Path): File path to the JSON environment configuration.

        Returns:
            None: Editor state is updated on success; error notification on failure.
        """
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
        """Create a new blank grid using values from width and height controls.

        Returns:
            None: Current grid state is replaced and save path is cleared.
        """
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

    def _cmd_load(self) -> None:
        """Open a file picker and load a selected environment config.

        Returns:
            None: Selected file is loaded when user confirms a path.
        """
        if not self._ask_discard():
            return
        path_str = filedialog.askopenfilename(
            title="Load environment configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="configs/env",
        )
        if path_str:
            self._load_from_path(Path(path_str))

    def _cmd_save(self) -> None:
        """Save to the current path or fall back to Save As for new files.

        Returns:
            None: Current config is persisted if validation and IO succeed.
        """
        if self._save_path is None:
            self._cmd_save_as()
        else:
            self._save_to_path(self._save_path)

    def _cmd_save_as(self) -> None:
        """Open a file picker and save the current config to a new path.

        Returns:
            None: Selected destination is used to persist the current config.
        """
        path_str = filedialog.asksaveasfilename(
            title="Save environment configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="configs/env",
        )
        if path_str:
            self._save_to_path(Path(path_str))

    def _save_to_path(self, path: Path) -> None:
        """Validate and write the config to disk.

        Args:
            path (Path): Destination file path for the JSON configuration.

        Returns:
            None: On success updates save metadata; on failure emits notifications.
        """
        errors = self._validate()
        if errors:
            first = errors[0]
            extra = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
            self._notify(f"Cannot save — {first}{extra}", "error")
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._format_config_json(self._to_dict()), encoding="utf-8")
            self._save_path = path
            self._dirty = False
            self.title(f"GridWorld Environment Editor — {path.name}")
            self._status_var.set(f"Saved: {path}")
            self._notify(f"Saved to {path.name}", "ok")
        except Exception as exc:
            self._notify(f"Save failed: {exc}", "error")

    def _cmd_validate(self) -> None:
        """Run validation checks and show either the first error or summary.

        Returns:
            None: Validation feedback is shown in the notification bar.
        """
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
        """Reset every cell to EMPTY while preserving current dimensions.

        Returns:
            None: Grid is redrawn and marked as modified.
        """
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                self._grid[r][c] = CellType.EMPTY
        self._full_redraw()
        self._mark_dirty()
        self._notify("Grid cleared.", "info")

    def _cmd_resize(self) -> None:
        """Resize the grid and preserve content in the overlapping area.

        Returns:
            None: Grid dimensions and display are updated on valid input.
        """
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
        """Decide whether destructive actions should proceed.

        Returns:
            bool: Always ``True`` in this implementation.
        """
        return True

    def _on_close(self) -> None:
        """Handle window close requests.

        Returns:
            None: Destroys the Tk window if discard policy permits closing.
        """
        if self._ask_discard():
            self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line options for launching the editor.

    Returns:
        argparse.Namespace: Parsed CLI arguments with optional config path.
    """
    p = argparse.ArgumentParser(description="GridWorld environment editor.")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON configuration file to open on startup.",
    )
    return p.parse_args()


def main() -> None:
    """Run the editor application when invoked as a script.

    Returns:
        None: Blocks until the Tkinter main loop exits.
    """
    args = parse_args()
    app = EditorApp(initial_config_path=args.config)
    app.mainloop()


if __name__ == "__main__":
    main()
