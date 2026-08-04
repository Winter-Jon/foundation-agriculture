# AgriNet 数据工作流

Data 域负责准备 AgriNet 样本、调用教师数据 provider、把教师记录转换为学生 SFT，并在每个版本化边界执行验证。稳定入口是 `agrinet data`；上层调用方不直接导入 VLOOM registry。

## 查找实验

```bash
.venv/bin/agrinet data list
.venv/bin/agrinet data show data-generate-agrinet-vloom-teacher-v1
```

## 本地操作

```bash
.venv/bin/agrinet data prepare EXPERIMENT_ID
.venv/bin/agrinet data generate EXPERIMENT_ID
.venv/bin/agrinet data convert EXPERIMENT_ID
.venv/bin/agrinet data validate PATH --kind prepared
.venv/bin/agrinet data validate PATH --kind teacher
.venv/bin/agrinet data validate PATH --kind sft
.venv/bin/agrinet data validate ARTIFACT_DIR --kind bounded
```

当前有界准备基线登记为 `data-prepare-agrinet-bounded-contrast-v1`。它读取现行 `wiki/base.json` schema，以 seed 42 分层选择 4 个病害类和 4 个虫害类，并保证每条样本的 3 个负候选全部同域。这是确定性的迁移基线，不冒充旧 feature-based hard-negative 流程的复现；后者依赖的 pair 和 representative artifact 当前磁盘中不存在。

每个实验命令都支持重复传入 `--config-override key=value`。可以显式提供环境变量；否则 VLOOM adapter 通过 `~/.apikeys/bin/apikey env yunwu` 解密云雾凭据。解密后的值不会进入 YAML、命令预览、manifest 或日志。

如果 GPG 提示私钥尚未解锁，请先在交互式终端中解锁一次且不要打印密钥，然后重试 AgriNet 命令：

```bash
~/.apikeys/bin/apikey env yunwu >/dev/null
```

当前环境没有显式代理变量时，VLOOM adapter 还会加载交互式 shell 中已有的 `proxy_on` 定义。代理值会经过严格 URL 校验，并与凭据一样只在子进程内存中传递。

## 本地运行

通过统一入口预览、前台运行，或显式转为后台长任务：

```bash
.venv/bin/agrinet data submit EXPERIMENT_ID --operation generate --dry-run
.venv/bin/agrinet data submit EXPERIMENT_ID --operation generate
.venv/bin/agrinet data submit EXPERIMENT_ID --operation generate --detach
```

前台和后台运行都会在 `outputs/runs/data/<experiment-id>/<run-id>/` 保存 `manifest.yaml`、`config.resolved.yaml` 和 `status.json`。后台运行额外把 stdout/stderr 保存到 `logs/`；前台输出保持连接当前终端。数据 artifact 是实验配置定义的另一类程序输出。现有 `slurm/` 文件属于迁移前历史。

## 数据契约

- 准备后样本：`agrinet.data.sample/v1`
- 教师记录：`agrinet.data.teacher/v1`
- 学生 SFT：`agrinet.sft.student/v1`

VLOOM 专属依赖被限制在 `agrinet.data.providers.vloom`。在生产 teacher 输出转换完成验证前，迁移前的旧脚本暂时保留；新调用方应使用 `agrinet data`。

VLOOM 从 `vendor/VLooM` 安装；来源以及尚待确认的上游许可证状态记录在 `vendor/VLooM/README.agrinet.md`。
