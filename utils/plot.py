


sns.set_theme(style="whitegrid", font="DejaVu Sans", rc={
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
})

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _maybe_to_numpy(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    return np.asarray(x)

def plot_training_curves_from_csvlog(csv_log_dir, save_dir):
    try:
        version_dirs = [d for d in os.listdir(csv_log_dir) if d.startswith("version_")]
        if not version_dirs:
            return
        latest = sorted(version_dirs, key=lambda x: int(x.split("_")[-1]))[-1]
        metrics_csv = os.path.join(csv_log_dir, latest, "metrics.csv")
        if not os.path.exists(metrics_csv):
            return
        df = pd.read_csv(metrics_csv)
        if "epoch" not in df.columns:
            return
        piv = df.pivot_table(index="epoch", columns="metric", values="value", aggfunc="mean")
        _ensure_dir(save_dir)

        plt.figure(figsize=(6,4))
        for col in piv.columns:
            if "loss" in str(col):
                plt.plot(piv.index, piv[col], label=col)
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training/Validation Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "curve_loss.png")); plt.close()

        acc_like = [c for c in piv.columns if ("acc" in str(c).lower() or "accuracy" in str(c).lower())]
        if acc_like:
            plt.figure(figsize=(6,4))
            for col in acc_like:
                plt.plot(piv.index, piv[col], label=col)
            plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Accuracy-like Curves")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "curve_acc.png")); plt.close()
    except Exception as e:
        print("[warn] plot_training_curves_from_csvlog failed:", e)

def _wrap_labels(labels, max_width=16):
    return [textwrap.fill(str(x), width=max_width, break_long_words=False) for x in labels]

def classification_report_figures(
    y_true,
    y_pred,
    class_names,
    save_path,                 
    page_rows=15,              
    col_width=2.2,             
    row_height=0.7,            
    font_size=11,              
    title_font_size=16,        
    cmap="YlGnBu",             
    class_name_wrap=16,        
    save_pdf=True              
):

    rep = classification_report(
        y_true, y_pred,
        target_names=[str(c) for c in class_names],
        output_dict=True, zero_division=0
    )
    df = pd.DataFrame(rep).T


    tail_keys = ["accuracy", "macro avg", "weighted avg"]
    tail_idx = [i for i, idx in enumerate(df.index) if str(idx) in tail_keys]
    if tail_idx:
        split_pos = min(tail_idx)
        df_class = df.iloc[:split_pos, :].copy()
        df_overall = df.iloc[split_pos:, :].copy()
    else:
        df_class = df.copy()
        df_overall = pd.DataFrame(columns=df.columns)


    wrapped_names = _wrap_labels(df_class.index.tolist(), max_width=class_name_wrap)
    df_class.index = wrapped_names

    n_classes = df_class.shape[0]
    n_pages = math.ceil(n_classes / page_rows) if n_classes > 0 else 0
    base, ext = os.path.splitext(save_path)
    out_paths = []

    for p in range(n_pages):
        sl = slice(p * page_rows, min((p + 1) * page_rows, n_classes))
        page_df = df_class.iloc[sl, :].round(3)

        rows = page_df.shape[0]
        cols = page_df.shape[1]

        fig_w = max(8.0, cols * col_width + 2.0)
        fig_h = max(6.0, rows * row_height + 2.0)

        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            page_df,
            annot=True, fmt=".3f",
            cmap=cmap, cbar=False,
            annot_kws={"size": font_size},
            linewidths=0.6, linecolor="#DDDDDD"
        )
        plt.title(f"Classification Report (Per-Class) — Page {p+1}/{n_pages}",
                  fontsize=title_font_size, pad=14)
        plt.xlabel("Metrics", fontsize=font_size+1)
        plt.ylabel("Classes", fontsize=font_size+1)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plt.tight_layout()
        png_path = f"{base}_perclass_page{p+1}.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        if save_pdf:
            pdf_path = f"{base}_perclass_page{p+1}.pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
            out_paths.append(pdf_path)
        out_paths.append(png_path)
        plt.close()


    if not df_overall.empty:
        page_df = df_overall.round(3)
        rows = page_df.shape[0]
        cols = page_df.shape[1]
        fig_w = max(6.5, cols * (col_width*0.9) + 1.5)
        fig_h = max(3.8, rows * (row_height*1.1) + 1.2)

        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            page_df,
            annot=True, fmt=".3f",
            cmap="Oranges", cbar=False,
            annot_kws={"size": font_size+1},
            linewidths=0.6, linecolor="#DDDDDD"
        )
        plt.title("Classification Report — Overall Summary", fontsize=title_font_size, pad=10)
        plt.xlabel("Metrics", fontsize=font_size+1)
        plt.ylabel("Summary", fontsize=font_size+1)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plt.tight_layout()
        png_path = f"{base}_overall.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        if save_pdf:
            pdf_path = f"{base}_overall.pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
            out_paths.append(pdf_path)
        out_paths.append(png_path)
        plt.close()

    return out_paths

def plot_confusions(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(6.5,5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Counts)")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "cm_counts.png")); plt.close()

    plt.figure(figsize=(6.5,5.5))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Purples", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "cm_normalized.png")); plt.close()

def plot_roc_pr_curves(y_true, y_proba, class_names, save_dir):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    _ensure_dir(save_dir)

    # --- ROC ---
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        if y_true_bin[:, i].sum() == 0:  # 该类未出现在 test
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro
    fpr_grid = np.linspace(0, 1, 1000)
    tprs = []
    for k in fpr:
        if k in ("micro", "macro"):
            continue
        tprs.append(np.interp(fpr_grid, fpr[k], tpr[k]))
    tprs = np.array(tprs)
    tpr_mean = tprs.mean(axis=0) if len(tprs)>0 else np.zeros_like(fpr_grid)
    roc_auc["macro"] = auc(fpr_grid, tpr_mean)

    plt.figure(figsize=(7,5))
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-avg ROC (AUC={roc_auc['micro']:.3f})", lw=2)
    plt.plot(fpr_grid, tpr_mean, label=f"macro-avg ROC (AUC={roc_auc['macro']:.3f})", lw=2)
    for i in range(n_classes):
        if i in roc_auc:
            plt.plot(fpr[i], tpr[i], lw=1, alpha=0.6, label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_multi.png")); plt.close()

    # --- PR ---
    plt.figure(figsize=(7,5))
    ap_micro = average_precision_score(y_true_bin, y_proba, average="micro")
    plt.plot([0,1],[ap_micro, ap_micro], "k--", lw=1, label=f"micro-avg AP={ap_micro:.3f}")
    for i in range(n_classes):
        if y_true_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, lw=1.5, label=f"{class_names[i]} (AP={ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_multi.png")); plt.close()

def plot_topk_and_calibration(y_true, y_proba, save_dir, ks=(1,2,3,5)):
    _ensure_dir(save_dir)
    n_classes = y_proba.shape[1]
    # Top-K
    vals = []
    for k in ks:
        k = min(k, n_classes)
        vals.append(top_k_accuracy_score(y_true, y_proba, k=k, labels=np.arange(n_classes)))
    plt.figure(figsize=(5.5,4))
    plt.plot(list(ks), vals, marker="o")
    plt.xlabel("k"); plt.ylabel("Top-k Accuracy"); plt.title("Top-k Accuracy Curve")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "topk.png")); plt.close()

    y_pred_conf = y_proba.max(axis=1)
    y_pred_cls  = y_proba.argmax(axis=1)
    correct     = (y_pred_cls == y_true).astype(int)
    prob_true, prob_pred = calibration_curve(correct, y_pred_conf, n_bins=15, strategy="quantile")
    plt.figure(figsize=(5.5,4))
    plt.plot([0,1],[0,1],"k--", alpha=0.5)
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted confidence"); plt.ylabel("Empirical accuracy")
    plt.title("Reliability Diagram"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration.png")); plt.close()

def plot_support_bars(y_true, class_names, save_path):
    cnts = pd.Series(y_true).value_counts().sort_index()
    plt.figure(figsize=(6,3.6))
    sns.barplot(x=class_names, y=cnts.values, color=sns.color_palette("Blues")[2])
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Count"); plt.title("Class Support (Test)")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_pca_embedding(preprocessor, X, y_true, y_pred, class_names, save_dir):
    _ensure_dir(save_dir)
    try:
        X_t = preprocessor.transform(X)   
        X_t = np.asarray(X_t)
        n = min(8000, X_t.shape[0])      
        idx = np.random.choice(X_t.shape[0], n, replace=False) if X_t.shape[0] > n else np.arange(X_t.shape[0])
        X_sub = X_t[idx]; y_t = np.asarray(y_true)[idx]; y_p = np.asarray(y_pred)[idx]
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(X_sub)

        plt.figure(figsize=(6.2,5))
        sc = plt.scatter(Z[:,0], Z[:,1], c=y_t, s=10, cmap="tab20", alpha=0.75)
        plt.colorbar(sc, ticks=range(len(class_names)))
        plt.title("PCA (colored by True)"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pca_true.png")); plt.close()

        plt.figure(figsize=(6.2,5))
        sc = plt.scatter(Z[:,0], Z[:,1], c=y_p, s=10, cmap="tab20", alpha=0.75)
        plt.colorbar(sc, ticks=range(len(class_names)))
        plt.title("PCA (colored by Pred)"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pca_pred.png")); plt.close()
    except Exception as e:
        print("[warn] PCA embedding failed:", e)

def save_artifacts(classifier, class_names, save_dir):
    _ensure_dir(save_dir)
    try:
        net = classifier.task_model.estimator
        torch.save(net.state_dict(), os.path.join(save_dir, "model_weights.pt"))
    except Exception as e:
        print("[warn] saving model state_dict failed:", e)

    bundle = {
        "config": getattr(classifier, "config", None),
        "preprocessor": getattr(classifier, "preprocessor", None),
        "class_names": class_names,
    }
    joblib.dump(bundle, os.path.join(save_dir, "bundle.joblib"))

    try:
        with open(os.path.join(save_dir, "training_params.json"), "w", encoding="utf-8") as f:
            json.dump(classifier.get_params(), f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print("[warn] saving training_params.json failed:", e)

def full_evaluate_and_visualize(classifier, X_train, y_train_enc, X_test, y_test_enc, class_names, out_dir="artifacts"):
    _ensure_dir(out_dir)

    y_pred = classifier.predict(X_test)
    try:
        y_proba = classifier.predict_proba(X_test)
    except Exception:

        n_classes = len(class_names)
        y_proba = np.eye(n_classes)[y_pred]


    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    plot_confusions(y_test_enc, y_pred, class_names, out_dir)
    classification_report_figures(y_test_enc, y_pred, class_names, os.path.join(out_dir, "classification_report.png"))
    plot_roc_pr_curves(y_test_enc, y_proba, class_names, out_dir)
    plot_topk_and_calibration(y_test_enc, y_proba, out_dir)
    plot_support_bars(y_test_enc, class_names, os.path.join(out_dir, "class_support.png"))
    plot_pca_embedding(classifier.preprocessor, X_test, y_test_enc, y_pred, class_names, out_dir)

    csv_log_dir = os.path.join(out_dir, "csv_logs")
    if os.path.isdir(csv_log_dir):
        plot_training_curves_from_csvlog(csv_log_dir, out_dir)

    save_artifacts(classifier, class_names, out_dir)

    print(f"[OK] 全部图与文件已输出到：{os.path.abspath(out_dir)}")
