"""Marks the fixture's ``models/`` as a real package.

Without it ``models`` is a namespace portion, and a regular package of the same
name found later on ``sys.path`` (backend has its own ``models/``) wins the
import outright — so ``models.resid`` resolves to the backend's package and the
worker cannot load the fixture at all. Real repositories ship this file; the
fixture must too, or it tests an import path no user has.
"""
