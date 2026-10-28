"""Synthetic Document Finetuning pipeline.

Deliberately empty of imports. `sdf.models.provider` is an `inspect_ai` entry point,
so this module is imported by every `inspect` and `scout` invocation on the machine;
anything expensive here is paid on each one.
"""
