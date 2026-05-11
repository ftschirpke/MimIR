import mim_py as mim
from pathlib import Path

from mim_py.mim_regex import MimRegex, RegBuilder
from mim_py.mim_enums.regex_plug import regex

repo_root = Path(__file__).resolve().parents[4]
plugin_dir = repo_root / "build" / "lib" / "mim"
if not plugin_dir.is_dir():
    raise FileNotFoundError(f"expected plugin directory at '{plugin_dir}'")

d = mim.Driver()
d.add_search_path(plugin_dir)
b = RegBuilder(d, "regex", mim.Level.Error)


alpha = b.range('a', 'z') | b.range('A', 'Z')                # [a-zA-Z]
under_hyph = b.lit("_") | b.lit("-")                         # [_\-]
dot_under_hyph = b.lit(".").disj([b.lit("_"), b.lit("-")])   # [._\-]
an = b.alnum()
# Local: [a-zA-Z0-9](?:[a-zA-Z0-9]*[._\-]+[a-zA-Z0-9])*[a-zA-Z0-9]*
local = (an
         + (an["*"] + dot_under_hyph["+"] + an)["*"]
         + an["*"])

# Domain: [a-zA-Z0-9](?:[a-zA-Z0-9]*[_\-]+[a-zA-Z0-9])*[a-zA-Z0-9]*
domain = (an
          + (an["*"] + under_hyph["+"] + an)["*"]
          + an["*"])

# Subdomain: (?:(?:[a-zA-Z0-9]*[_\-]+[a-zA-Z0-9])*[a-zA-Z0-9]+\.)*
subdomain = ((an["*"] + under_hyph["+"] + an)["*"]
             + an["+"]
             + b.lit("."))["*"]

# Top Level Domain (.com, .de, etc...): [a-zA-Z][a-zA-Z]+
tld = alpha + alpha["+"]

# Zusammengefasster Ausdruck
email = local + b.lit("@") + domain + b.lit(".") + subdomain + tld
library = email.jit()

# print(x.regex)
# print(library.items())
if(library["match_func"]("test@test.de".encode("utf-8"))):
    print("python package works")
