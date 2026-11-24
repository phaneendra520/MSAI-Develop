"""
CineMatch conversational agent (Extension E6: natural-language interface).

A rule-based, intent-driven agent: no LLM is required, so every response is
traceable to the recommender. It keeps a per-session Profile and settings, and
understands requests such as

    I loved Toy Story and The Matrix
    I didn't like Titanic
    something like Alien
    recommend 5 comedies from the 90s
    more variety / less variety / hidden gems / more popular
    why Fargo?
    show my profile / reset / help

Run the text-prompt version:  python agent.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from recommender import CineMatch, Profile, get_engine

GENRE_WORDS = {
    "action": "Action", "adventure": "Adventure", "animation": "Animation", "animated": "Animation",
    "cartoon": "Animation", "kids": "Children's", "children": "Children's", "family": "Children's",
    "comedy": "Comedy", "comedies": "Comedy", "funny": "Comedy", "crime": "Crime",
    "documentary": "Documentary", "documentaries": "Documentary", "drama": "Drama", "dramas": "Drama",
    "fantasy": "Fantasy", "noir": "Film-Noir", "horror": "Horror", "scary": "Horror",
    "musical": "Musical", "musicals": "Musical", "mystery": "Mystery", "romance": "Romance",
    "romantic": "Romance", "sci-fi": "Sci-Fi", "scifi": "Sci-Fi", "science fiction": "Sci-Fi",
    "thriller": "Thriller", "thrillers": "Thriller", "war": "War", "western": "Western",
    "westerns": "Western",
}

HELP = (
    "I'm CineMatch. Tell me movies you like and I'll find more. Try:\n"
    "  • I loved Toy Story and The Matrix\n"
    "  • I didn't like Titanic\n"
    "  • something like Alien\n"
    "  • recommend 5 comedies from the 90s\n"
    "  • more variety | less variety | hidden gems | more popular\n"
    "  • why The Fugitive?\n"
    "  • profile | reset | help"
)


@dataclass
class Session:
    profile: Profile = field(default_factory=Profile)
    genres: list = field(default_factory=list)
    year_range: tuple | None = None
    diversity: float = 0.3
    w_pop: float = 0.1
    last: list = field(default_factory=list)       # last recommendation dicts
    shown: set = field(default_factory=set)        # titles already shown ("more" paging)


class CineMatchAgent:
    def __init__(self, engine: CineMatch | None = None):
        self.e = engine or get_engine()
        self.s = Session()

    # ------------------------------------------------------------ parsing helpers
    def _titles(self, text):
        """Split a free-text list of titles and resolve each with fuzzy search."""
        quoted = re.findall(r'"([^"]+)"', text)
        parts = quoted or re.split(r",|\band\b|\bplus\b|&|;", text)
        found, missed = [], []
        for p in parts:
            p = re.sub(r"\b(the movie|movie|film|films|movies|a lot|so much|really|too|also)\b", "", p,
                       flags=re.I).strip(" .!?")
            if len(p) < 2:
                continue
            hit = self.e.find(p, n=1)
            (found if hit else missed).append(hit[0] if hit else p)
        return found, missed

    @staticmethod
    def _genres(text):
        t = text.lower()
        return sorted({g for w, g in GENRE_WORDS.items() if re.search(rf"\b{re.escape(w)}\b", t)})

    @staticmethod
    def _years(text):
        t = text.lower()
        m = re.search(r"\b(?:the\s+)?(19|20)?(\d0)'?s\b", t)
        if m:
            century = m.group(1) or ("20" if m.group(2) in ("00", "10") else "19")
            start = int(century + m.group(2))
            return (start, start + 9)
        m = re.search(r"\b(after|since)\s+((?:19|20)\d{2})", t)
        if m:
            return (int(m.group(2)), 2000)
        m = re.search(r"\bbefore\s+((?:19|20)\d{2})", t)
        if m:
            return (1900, int(m.group(1)) - 1)
        m = re.search(r"\b((?:19|20)\d{2})\s*(?:-|to)\s*((?:19|20)\d{2})\b", t)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return None

    @staticmethod
    def _count(text, default=5):
        m = re.search(r"\b(\d{1,2})\b(?!\d)(?!'?s)", text)
        if m and not re.search(r"\b(19|20)\d{2}\b", m.group(0)):
            n = int(m.group(1))
            if 1 <= n <= 20:
                return n
        return default

    # ------------------------------------------------------------ main entry
    def respond(self, text: str) -> str:
        raw = text.strip()
        t = raw.lower()
        if not t:
            return "Tell me a movie you enjoyed, or type 'help'."

        if t in ("help", "?", "what can you do", "commands"):
            return HELP
        if re.match(r"^(hi|hello|hey)\b", t) and len(t) < 12:
            return "Hi! Name a few movies you love and I'll recommend more. (Type 'help' for examples.)"
        if t in ("reset", "start over", "clear"):
            self.s = Session()
            return "Done, your profile and filters are cleared. What do you like to watch?"
        if t.startswith(("profile", "show my profile", "what do you know")):
            return self._profile_text()
        if t.startswith(("clear filter", "no filter", "any genre", "remove filter")):
            self.s.genres, self.s.year_range = [], None
            return "Filters removed.\n\n" + self._recommend(5)

        m = re.match(r"^(why|explain)\s+(.*)", t)
        if m:
            return self._why(raw[len(m.group(1)):].strip(" ?"))

        if re.search(r"\b(more variety|more diverse|mix it up|surprise me|different)\b", t):
            self.s.diversity = min(0.8, self.s.diversity + 0.25)
            return f"Turning variety up to {self.s.diversity:.2f}.\n\n" + self._recommend(5, fresh=True)
        if re.search(r"\b(less variety|less diverse|more similar|stay focused|similar ones)\b", t):
            self.s.diversity = max(0.0, self.s.diversity - 0.25)
            return f"Turning variety down to {self.s.diversity:.2f}.\n\n" + self._recommend(5, fresh=True)
        if re.search(r"\b(hidden gems?|obscure|lesser.known|underrated)\b", t):
            self.s.w_pop = 0.0
            return ("Okay, here are less-watched titles that still fit your taste.\n\n"
                    + self._recommend(5, fresh=True, min_votes=20, hidden=True))
        if re.search(r"\b(more popular|mainstream|crowd.pleas|blockbuster)\b", t):
            self.s.w_pop = 0.4
            return "Leaning toward crowd favourites.\n\n" + self._recommend(5, fresh=True)

        m = re.search(r"\b(?:didn'?t like|did not like|don'?t like|do not like|hated?|dislike[d]?|not a fan of|can'?t stand)\b(.*)", t)
        if m:
            ids, missed = self._titles(raw[m.start(1):])
            for i in ids:
                self.s.profile.dislike(i)
            msg = ("Noted, I'll steer away from " + ", ".join(self.e.title(i) for i in ids) + ".") if ids else ""
            if missed:
                msg += f"\nI couldn't find: {', '.join(missed)}."
            return (msg + "\n\n" + self._recommend(5, fresh=True)).strip()

        m = re.search(r"\b(?:like|love[sd]?|liked|enjoy(?:ed)?|adore[d]?|fan of|favou?rites? (?:are|is)|watched)\b(.*)", t)
        sim = re.search(r"\b(?:something|movies?|films?|anything)\s+(?:like|similar to)\b(.*)", t)
        if sim or (m and m.group(1).strip()):
            seg = raw[(sim or m).start(1):]
            seg = re.split(r"\b(?:from the|in the|after|before|since)\b", seg, flags=re.I)[0]
            ids, missed = self._titles(seg)
            g, y = self._genres(t), self._years(t)
            if sim:                       # "something like X" = a fresh, focused request
                self.s.genres, self.s.year_range = g, y
            else:
                if g:
                    self.s.genres = g
                if y:
                    self.s.year_range = y
            if not ids:
                return f"I couldn't find {', '.join(missed) or 'that title'} in my catalog (MovieLens covers 1919-2000). Try another title?"
            for i in ids:
                self.s.profile.like(i)
            msg = "Got it, you like " + ", ".join(self.e.title(i) for i in ids) + "."
            if missed:
                msg += f" (I couldn't find: {', '.join(missed)}.)"
            return msg + "\n\n" + self._recommend(self._count(t), fresh=True, focus=ids if sim else None)

        if re.search(r"\b(recommend|suggest|what should i watch|show me|give me|find|more|next|another)\b", t) \
                or self._genres(t) or self._years(t):
            g, y = self._genres(t), self._years(t)
            fresh = bool(g or y) or not re.search(r"\b(more|next|another)\b", t)
            if g:
                self.s.genres = g
            if y:
                self.s.year_range = y
            return self._recommend(self._count(t), fresh=fresh)

        # Fallback: maybe the user just typed a title
        ids, _ = self._titles(raw)
        if ids:
            for i in ids:
                self.s.profile.like(i)
            return ("I'll take that as a movie you like: " + ", ".join(self.e.title(i) for i in ids)
                    + ".\n\n" + self._recommend(5, fresh=True))
        return "Sorry, I didn't catch that. " + HELP

    # ------------------------------------------------------------ actions
    def _recommend(self, k=5, fresh=False, min_votes=None, hidden=False, focus=None):
        if fresh:
            self.s.shown = set()
        exclude = {r["idx"] for r in self.s.last} if not fresh else set()
        exclude |= self.s.shown
        profile = self.s.profile
        if focus:   # "something like X": X dominates, the rest of the profile is a light tie-breaker
            profile = Profile({i: (w * 0.25 if w > 0 else w) for i, w in self.s.profile.items.items()})
            for i in focus:
                profile.items[i] = 1.0
        import numpy as np
        max_votes = float(np.percentile(self.e.vote_count, 75)) if hidden else None
        recs = self.e.recommend(profile, k=k, w_content=0.1, w_cf=0.8, w_pop=self.s.w_pop,
                                diversity=self.s.diversity, genres=self.s.genres or None,
                                year_range=self.s.year_range, exclude=exclude, min_votes=min_votes,
                                max_votes=max_votes)
        if not recs:
            return "I ran out of matches with these filters. Try 'clear filters' or name another movie."
        self.s.last = recs
        self.s.shown |= {r["idx"] for r in recs}
        head = "Here are some picks" if self.s.profile.items else "You haven't told me your taste yet, so here are top-rated crowd favourites"
        filt = []
        if self.s.genres:
            filt.append("/".join(self.s.genres))
        if self.s.year_range:
            filt.append(f"{self.s.year_range[0]}-{self.s.year_range[1]}")
        head += f" ({', '.join(filt)})" if filt else ""
        lines = [head + ":"]
        for n, r in enumerate(recs, 1):
            lines.append(f"{n}. {r['title']} [{r['genres']}], {r['match']}% match\n   Why: {r['why']}")
        lines.append("\nSay 'more', 'why [title]', 'I didn't like [title]', or 'more variety'.")
        return "\n".join(lines)

    def _why(self, text):
        ids = self.e.find(text, n=1)
        if not ids:
            return f"I couldn't find '{text}'."
        i = ids[0]
        for r in self.s.last:
            if r["idx"] == i:
                return f"{r['title']}: {r['why']} (match {r['match']}%)"
        rel, parts = self.e.score(self.s.profile)
        r = self.e.explain(i, self.s.profile, rel[i], parts)
        return f"{r['title']} isn't in your current list, but it would score {r['match']}%. {r['why']}"

    def _profile_text(self):
        likes = [self.e.title(i) for i, w in self.s.profile.items.items() if w > 0]
        dis = [self.e.title(i) for i, w in self.s.profile.items.items() if w < 0]
        return ("Your profile\n"
                f"  Likes: {', '.join(likes) or '(none yet)'}\n"
                f"  Dislikes: {', '.join(dis) or '(none)'}\n"
                f"  Genre filter: {', '.join(self.s.genres) or 'any'}\n"
                f"  Years: {self.s.year_range or 'any'}\n"
                f"  Variety: {self.s.diversity:.2f}   Popularity weight: {self.s.w_pop:.2f}")


def main():
    print("Loading CineMatch (MovieLens 1M)...")
    agent = CineMatchAgent()
    print(HELP)
    while True:
        try:
            text = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if text.strip().lower() in ("quit", "exit", "bye"):
            print("CineMatch: Enjoy the movies!")
            break
        print("\nCineMatch: " + agent.respond(text))


if __name__ == "__main__":
    main()
