"""Point-in-time tools. Every data access an agent makes goes through a registered tool whose availability
rule is declared (harness.pit.registry); the registry refuses calls that would reveal anything after the
decision date, and logs every call so the audit can replay it."""
