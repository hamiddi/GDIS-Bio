param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PipelineArgs)
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Interpreter = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
& $Interpreter (Join-Path $Root "tools/pipeline.py") @PipelineArgs
exit $LASTEXITCODE
