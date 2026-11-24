# 本地 Zotero 与论文引用同步

## 推荐：保留文献库，仅同步论文

本次删除的是正文 cite 调用和表格比较行，未修改 Zotero 自动导出的 references.bib，也未删除 Zotero 中的条目。main.tex 使用 references 与 references_extra 两个数据库，没有 nocite{*}；因此只有实际引用的条目进入论文参考文献。

在本地仓库拉取 master 后，进入 paper/ 重新编译：

```bash
git pull --ff-only origin master
cd paper
latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

正常重新编译会更新引用编号和参考文献。若 PDF 仍显示旧条目，先清理 BibTeX 缓存再重建：

```bash
latexmk -C -outdir=build main.tex
latexmk -xelatex -halt-on-error -file-line-error -outdir=build main.tex
```

保留 GFMR、ConceptBank、PRS-Seg 的 Zotero 条目和 bib 记录不会让它们重新进入参考文献。不要为了清理显示直接手改自动导出的 references.bib。

## 可选：同时整理 Zotero 的论文集合

1. 在 Zotero 的本论文集合中找到 GFMR、ConceptBank、PRS-Seg。ConceptBank 若有重复记录，一并检查集合归属。
2. 选择“从集合中移除”（Remove Item from Collection）；文献仍保留在“我的文库”，不必使用“移到回收站”。
3. 若现有 Better BibTeX 自动导出针对整个文库，移出集合不会缩减导出文件。希望只导出论文集合时，右击该集合，选择“导出分类／集合”，格式使用 Better BibTeX，勾选 Keep updated，输出到本地项目 paper/references.bib。
4. 已有自动导出正确指向此集合和路径时，沿用原任务；避免两个自动导出同时写同一个文件。导出会由 Zotero 集合内容触发更新，不会根据 LaTeX 引用自动删除条目。
5. 保持已经引用的 citation key 不变；若修改 key，须同步修改 tex 的 cite 调用。导出后检查 Git diff，再编译确认没有未定义引用。

references_extra.bib 目前保存人工补充的引用，继续保留。无需为同步删除操作把它整体重新导入 Zotero；若以后要统一迁入 Zotero，应先按 DOI／标题去重并保持 key，再从 extra 中删除已迁移的同 key 条目，避免两份数据库重复定义。

## 官方说明

- Zotero 集合与条目移除：https://www.zotero.org/support/collections_and_tags
- Better BibTeX 自动导出：https://retorque.re/zotero-better-bibtex/exporting/auto/
