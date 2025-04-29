# Add an Article

This assumes that the original draft of the article, plus its code, is in the `scratch` repo. It also assumes you have already generated the `.md` file via iPython, R Markdown, or Quarto.

0. Pull the repository **main branch only**.
```
git pull origin main
```
I know this is supposed to happen by default, but I've noticed that if I just do `git pull` it seems to pull the `gh-pages` branch too, which I don't want locally.

1. Copy the directory from `scratch` into `WVExamples`. You might want to commit/push this before continuing, so you have the correct github link to the directory.
2. Create a new folder in `website/content/blog` for the new article.
3. Copy the `.md` file and any images into the new folder. **_The images all have to be at the same level as the `md` file._** This means you have to pull any files in the `*_files` directory to the top level.
4. Add a new yaml header to the `.md` file., for example:

```
---
title: 'Partial Pooling for Lower Variance Variable Encoding'
author: "Nina Zumel"
date: 2025-01-08
tags: ["partial pooling", "R"]
source: https://github.com/WinVector/WVExamples/tree/main/PartialPooling_R
---
```

`date` is in YYYY-MM-DD format. If there is only one tag, you don't need the brackets. `source` is optional, but if you have it, it should point to the location of the code directory in `WVExamples`.

5. Fix the links to figures in the `.md`. If the figure is added to the document via html (rather than markdown), you need to add an `alt` tag. It's required. Also some symbols such as double open braces error-out the substitution engine with a cryptic error message (and wrong line number).

```
<img src="terraces.jpg" alt="Banaue rice terraces">
```

I think that the alt tag is automatically added to images that are included via markdown notation

6. To preview the site:
```
npm run start
```

This puts a local website at `localhost:8080` (or wherever). This will catch any coding errors, and will show you what the website will look like.

The post will take title/date/author from the yaml. If you added a source link to the yaml, it will add a pointer to that directory at the top of the post.

## Deploy the website

Once the article looks to your satisfaction:

1. Kill the watcher process that runs the previewer. 

2. Commit/push your changes.

3. Delete the `_site` directory in `website`. This is really a belt-and-suspenders move. It prevents some bad relative url mistakes.

4. Recreate `_site` with the correct root directory:
```
npm run build
```

(This sets the root directory of the website to `/WVExamples/` rather than `/`, which is what we want.)

5. Push the website to the `gh-pages` branch.
```
npm run deploy
```

This copies `_site` to the `gh-pages` branch, then commits/pushes it. The push triggers a github action to deploy the new website. You can see the progress of the github action by going to the `WVExamples` repo at github, then clicking on the **Actions** tab at the top.

Once the deployment is done, the website is live at
`winvector.github.io/WVExamples`.