import re


def e4m3_instructions(ptx):
    pattern = re.compile("\\b(?:mma\\.sync|wgmma\\.mma_async)(?:\\.[\\w]+)*[^;]*\\be4m3\\b")
    return sorted({line.strip() for line in ptx.splitlines() if pattern.search(line)})
