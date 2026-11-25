from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = collect_dynamic_libs("onnxruntime", search_patterns=["onnxruntime_providers_shared.dll"])
