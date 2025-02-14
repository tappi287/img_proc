def exception_and_traceback(e):
    try:
        return "\n".join(traceback.format_exception(e))
    except Exception as _e:
        return f"{_e}\n{e}"
