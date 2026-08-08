from __future__ import annotations


class RenderCancelled(Exception):
    """Raised out of `render_image` when its `should_cancel` hook says stop.

    Its own type rather than a bool return so a cancelled render cannot be
    mistaken for a finished one by a caller that forgot to check.
    """
