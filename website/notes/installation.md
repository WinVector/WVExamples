# Initial Installation

1. Install Node.js -  https://nodejs.org/en/download/.

This installs `npm`, which is what you need to run the site generator.

2. Clone the website repo **main branch only** - 
```
git clone --single-branch --branch main git@github.com:WinVector/WVExamples.git
```

There are two branches, `main` and `gh-pages`. `gh-pages` is the website branch, and it's kind of large (also you will have a copy in the  `_site` directory of `main`).

The directory `website` holds the "source code" for building the website, and the other directories should correspond to specific articles.

3. In the local copy of the repo, go to the directory `website`, then run

```
npm install
```

This creates the `node_modules` directory in `website`. It holds local copies of all the modules needed to generate the site.

If you've curious, the list of modules to be installed is in `package.json` (at the bottom). The "scripts" field shows the actual definitions of the scripts that we have to run via `npm run ...` (as shown below).

4. To build the site locally (for preview)
```
npm run start
```

This will put a local version of the website at `localhost:8080` (or whatever it tells you). Changes to the contents of the `website/content` directory will be reflected in the website.

To add an article to the website, see the **Add an Article** document.