x2paddle 好像不更新了，一堆算子都不支持，所以魔改下代码方便转模型

- 修正 `aten::max`
- 修正 `aten::min`
- 增加 `aten::ge`（`aten::le` 都有的你和我说没这个？？？ ）
- 增加 `aten::pad`