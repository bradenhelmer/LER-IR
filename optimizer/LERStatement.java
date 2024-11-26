package optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import org.antlr.v4.runtime.tree.*;
import org.w3c.dom.Node;

public class LERStatement {
    public enum ExpressionType {
        ADD, MUL, OPERAND
    }

    public enum OperandType {
        ARRAY, SCALAR, CONSTANT, INCREMENTAL
    }

    public enum LoopType {
        REGULAR, SUMMATION, WHILE
    }

    public ArrayList<LERStatement> optimize() {
        return null;
    }

    public static class Loop {
        public LoopType loopType;
        public String id;
        public String regloopid;
        public String oldloopid;
        public String lBound;
        public String uBound;
        public Loop leftLoop;
        public Loop rightLoop;

        public String toString() {
            if (loopType == LoopType.REGULAR) {
                return "Γ" + id + "∫" + lBound + "," + uBound + "∫";
            } else if (loopType == LoopType.SUMMATION) {
                return "Σ" + id + "∫" + lBound + "," + uBound + "∫";
            } else {
                return "Ψ" + "$" + id + "$1>0∫";
            }
        }

        public int getCost() {
            assert (loopType != LoopType.WHILE);
            try {
                return Integer.parseInt(uBound) - Integer.parseInt(lBound);
            } catch (Exception e) {
                return 0;
            }
        }
    }

    public class Operand {
        public OperandType operandType;
        public String id;
        public int value = -1;
        public String dependence;
        public ParseTree index;
        public ArrayList<String> abstract_index;
        public ArrayList<String> encapsulated_abstract_index;
        public String flagged_id;
        public String new_id;

        public String toString() {
            if (operandType == OperandType.CONSTANT) {
                return Integer.toString(value);
            } else if (operandType == OperandType.SCALAR) {
                if (dependence == null)
                    return id;
                else
                    return id + "$" + dependence + "$";
            } else if (operandType == OperandType.ARRAY) {
                String idxStr;
                if (index != null) {
                    idxStr = "[" + index.getText() + "]";
                } else {
                    idxStr = abstract_index.toString();
                    // Remove spaces
                    idxStr = idxStr.replaceAll("\\s+", "");
                }
                if (dependence == null)
                    return id + idxStr;
                else
                    return id + "$" + dependence + "$" + idxStr;
            } else {
                String abst_repl;
                if (value != -1) {
                    abst_repl = "[" + new_id + "]";
                } else {
                    abst_repl = abstract_index.toString();
                }
                // Replace flagged_id with new_id
                if (flagged_id != null) {
                    abst_repl = abst_repl.replaceAll(flagged_id, new_id);
                } else {
                    System.out.println("Error: flagged_id is null");
                    assert (false);
                }
                String inclVar = "incl" + abst_repl;
                String incStr = "(" + inclVar + ":incl" + abst_repl.replaceAll(new_id, new_id + "-1");
                if (value != -1) {
                    incStr += "+" + value;
                    incStr += "->incl" + abst_repl.replaceAll(new_id, "0") + "=" + value + ")";
                } else {
                    incStr += "+" + id + abst_repl;
                    incStr += "->incl" + abst_repl.replaceAll(new_id, "0") + "=" + id
                            + abst_repl.replaceAll(new_id, "0") + ")";
                }
                return incStr;
            }
        }

        public String toAbstractString() {
            if (operandType == OperandType.CONSTANT) {
                return Integer.toString(value);
            } else if (operandType == OperandType.SCALAR) {
                if (dependence == null)
                    return id;
                else
                    return id + "$" + dependence + "$";
            } else if (operandType == OperandType.ARRAY) {
                String idxStr;
                idxStr = abstract_index.toString();
                // Remove spaces
                idxStr = idxStr.replaceAll("\\s+", "");
                // Replace [] with {}
                idxStr = idxStr.replace("[", "{");
                idxStr = idxStr.replace("]", "}");
                if (dependence == null)
                    return id + idxStr;
                else
                    return id + "$" + dependence + "$" + idxStr;
            } else {
                String abst_repl;
                if (value != -1) {
                    abst_repl = "[" + new_id + "]";
                } else {
                    abst_repl = abstract_index.toString();
                }
                // Replace flagged_id with new_id
                if (flagged_id != null) {
                    abst_repl = abst_repl.replaceAll(flagged_id, new_id);
                } else {
                    System.out.println("Error: flagged_id is null");
                    assert (false);
                }
                String inclVar = "incl" + abst_repl;
                String incStr = "(" + inclVar + ":incl" + abst_repl.replaceAll(new_id, new_id + "-1");
                if (value != -1) {
                    incStr += "+" + value;
                    incStr += "->incl" + abst_repl.replaceAll(new_id, "0") + "=" + value + ")";
                } else {
                    incStr += "+" + id + abst_repl;
                    incStr += "->incl" + abst_repl.replaceAll(new_id, "0") + "=" + id
                            + abst_repl.replaceAll(new_id, "0") + ")";
                }
                return incStr;
            }
        }

        public String toEncapsulatedString() {
            if (operandType == OperandType.CONSTANT) {
                return Integer.toString(value);
            } else if (operandType == OperandType.SCALAR) {
                if (dependence == null)
                    return id;
                else
                    return id + "$" + dependence + "$";
            } else if (operandType == OperandType.ARRAY) {
                String idxStr;
                idxStr = encapsulated_abstract_index.toString();
                // Remove spaces
                idxStr = idxStr.replaceAll("\\s+", "");
                // Replace [] with {}
                idxStr = idxStr.replace("[", "{");
                idxStr = idxStr.replace("]", "}");
                if (dependence == null)
                    return id + idxStr;
                else
                    return id + "$" + dependence + "$" + idxStr;
            } else {
                return " NOPE THIS IS A BAD SPOT TO BE IN";
            }
        }
    }

    public String toString() {
        String res = "";
        for (Loop l : L) {
            res += l.toString();
        }
        for (Operand e : E) {
            res += e.toString();
            // If the operand is not the last one, add either a + or *
            if (E.indexOf(e) != E.size() - 1) {
                if (EType == ExpressionType.ADD) {
                    res += "+";
                } else if (EType == ExpressionType.MUL) {
                    res += "*";
                }
            }
        }
        res += "=" + R.toString();
        return res;
    }

    public String toAbstractString() {
        String res = "";
        for (Loop l : L) {
            res += l.toString();
        }
        for (Operand e : E) {
            res += e.toAbstractString();
            // If the operand is not the last one, add either a + or *
            if (E.indexOf(e) != E.size() - 1) {
                if (EType == ExpressionType.ADD) {
                    res += "+";
                } else if (EType == ExpressionType.MUL) {
                    res += "*";
                }
            }
        }
        res += "=" + R.toString();
        return res;
    }

    public String toEncapsulatedString() {
        String res = "";
        for (Loop l : EncapsulatedLoops) {
            res += l.toString();
        }
        for (Operand e : E) {
            res += e.toEncapsulatedString();
            // If the operand is not the last one, add either a + or *
            if (E.indexOf(e) != E.size() - 1) {
                if (EType == ExpressionType.ADD) {
                    res += "+";
                } else if (EType == ExpressionType.MUL) {
                    res += "*";
                }
            }
        }
        res += "=" + R.toString();
        return res;
    }

    public void encapsulateLoops() {
        try {
            Loop prevLoop = null;
            HashSet<String> encapsulated_set = new HashSet<String>();
            for (Loop l : L) {
                if (prevLoop != null) {
                    // System.out.println("Evaluating loop: " + l.toString());
                    if (l.loopType == LoopType.SUMMATION || prevLoop.loopType == LoopType.SUMMATION) {
                        // System.out.println("One summation loop: " + l.toString());
                        if (l.uBound.equals(prevLoop.id)) {
                            Loop newLoop = new Loop();
                            newLoop.loopType = LoopType.SUMMATION;
                            newLoop.id = "t";
                            if (l.loopType == LoopType.REGULAR) {
                                newLoop.regloopid = l.id;
                                newLoop.oldloopid = prevLoop.id;
                            } else {
                                newLoop.regloopid = prevLoop.id;
                                newLoop.oldloopid = l.id;
                            }
                            newLoop.leftLoop = prevLoop;
                            newLoop.rightLoop = l;
                            newLoop.lBound = Integer
                                    .toString(Math.min(Integer.parseInt(prevLoop.lBound), Integer.parseInt(l.lBound)));
                            newLoop.uBound = Integer
                                    .toString(Integer.parseInt(prevLoop.uBound) * Integer.parseInt(prevLoop.uBound));
                            encapsulated_set.add(prevLoop.id);
                            encapsulated_set.add(l.id);
                            // System.out.println("Adding encapsulated loop: " + newLoop.toString());
                            // EncapsulatedLoops.add(newLoop);
                            prevLoop = newLoop;
                            continue;
                        } else {
                            EncapsulatedLoops.add(prevLoop);
                        }
                    } else {
                        EncapsulatedLoops.add(prevLoop);
                    }
                }
                prevLoop = l;
            }
            EncapsulatedLoops.add(prevLoop);
            // Populate encapsulated_abstract_index for each operand in E
            for (Operand e : E) {
                if (e.operandType == OperandType.ARRAY) {
                    HashSet<String> tmp_encapsulated_set = new HashSet<String>();
                    for (String i : e.abstract_index) {
                        if (encapsulated_set.contains(i)) {
                            tmp_encapsulated_set.add("t");
                        } else {
                            tmp_encapsulated_set.add(i);
                        }
                    }
                    e.encapsulated_abstract_index = new ArrayList<String>(tmp_encapsulated_set);
                }
            }
        } catch (Exception e) {
            System.out.println("Error in encapsulateLoops: " + e);
            e.printStackTrace();
        }
    }

    HashSet<String> relLoops(Loop l) {
        HashSet<String> res = new HashSet<String>();
        try {
            String loopid = l.id;
            for (Operand o : E) {
                if (o.operandType == OperandType.ARRAY) {
                    for (String i : o.encapsulated_abstract_index) {
                        if (i.equals(loopid)) {
                            res.addAll(o.encapsulated_abstract_index);
                            if (o.dependence != null) {
                                res.add(o.dependence);
                            }
                        }
                    }
                    if (o.dependence != null) {
                        if (o.dependence.equals(loopid)) {
                            res.addAll(o.encapsulated_abstract_index);
                            res.add(o.dependence);
                        }
                    }
                }
                if (o.operandType == OperandType.SCALAR) {
                    if (o.dependence != null) {
                        if (o.dependence.equals(loopid)) {
                            res.add(o.dependence);
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("Error in relLoops: " + e);
            e.printStackTrace();
        }
        return res;
    }

    public class LoopNode {
        public LoopNode parent = null;
        public Loop node;
        public int cost;
        public HashSet<String> relloopids;
        public ArrayList<LoopNode> children;
        public Operand result;

        public LoopNode(HashSet<String> relloopids, int cost, Loop node) {
            this.relloopids = new HashSet<String>(relloopids);
            this.cost = cost;
            children = new ArrayList<LoopNode>();
            this.node = node;
            result = null;
        }

        public void addParent(LoopNode parent) {
            this.parent = parent;
            parent.children.add(this);
        }

        public void confirmParent() {
            ArrayList<LoopNode> toRemove = new ArrayList<>();
            for (LoopNode child : children) {
                if (child.parent == null) {
                    child.parent = this;
                } else {
                    toRemove.add(child);
                }
            }
            children.removeAll(toRemove);
        }
    }

    public ArrayList<LoopNode> generateWorklist() {
        ArrayList<LoopNode> worklist = new ArrayList<LoopNode>();
        try {
            for (Loop l : EncapsulatedLoops) {
                if (l.loopType == LoopType.SUMMATION) {
                    HashSet<String> relevantloops = relLoops(l);
                    int cost = l.getCost();
                    LoopNode node = new LoopNode(relevantloops, cost, l);
                    worklist.add(node);
                }
            }
        } catch (Exception e) {
            System.out.println("Error in generateWorklist: " + e);
            e.printStackTrace();
        }
        return worklist;
    }

    LoopNode getMinimumCost(ArrayList<LoopNode> worklist) {
        int minCost = Integer.MAX_VALUE;
        LoopNode minNode = null;
        for (LoopNode node : worklist) {
            if (node.cost < minCost) {
                minCost = node.cost;
                minNode = node;
            }
        }
        return minNode;
    }

    public ArrayList<LoopNode> getForestRoots(ArrayList<LoopNode> forest) {
        ArrayList<LoopNode> roots = new ArrayList<LoopNode>();
        for (LoopNode l : forest) {
            if (l.parent == null) {
                roots.add(l);
            }
        }
        return roots;
    }

    public void printTree(LoopNode node, int depth) {
        for (int i = 0; i < depth; i++) {
            System.out.print("  ");
        }
        System.out.println("{" + node.node.id + ": " + node.relloopids.toString() + " (" + node.cost + ")}");
        for (LoopNode l : node.children) {
            printTree(l, depth + 1);
        }
    }

    public void printForest(ArrayList<LoopNode> forest) {
        System.out.println("------  Forest Output  ------");
        for (LoopNode l : getForestRoots(forest)) {
            printTree(l, 0);
        }
    }

    public ArrayList<Operand> getAvailOperands() {
        ArrayList<Operand> availOperands = new ArrayList<Operand>();
        for (Operand e : E) {
            availOperands.add(e);
        }
        return availOperands;
    }

    public ArrayList<LoopNode> getDepthListHelper(LoopNode node, int depth, int depthCount) {
        ArrayList<LoopNode> res = new ArrayList<LoopNode>();
        if (depthCount == depth) {
            res.add(node);
        }
        if (node.children.isEmpty()) {
            return res;
        } else {
            for (LoopNode l : node.children) {
                res.addAll(getDepthListHelper(l, depth, depthCount + 1));
            }
        }
        return res;
    }

    public ArrayList<LoopNode> getDepthList(LoopNode node, int depth) {
        ArrayList<LoopNode> res = new ArrayList<LoopNode>();
        int depthCount = 0;
        if (depthCount == depth) {
            res.add(node);
        }
        if (node.children.isEmpty()) {
            return res;
        } else {
            for (LoopNode l : node.children) {
                res.addAll(getDepthListHelper(l, depth, depthCount + 1));
            }
        }
        return res;
    }

    public int getMaxDepthHelper(LoopNode node, int depth) {
        if (node.children.isEmpty()) {
            return depth;
        }
        int maxDepth = 0;
        for (LoopNode l : node.children) {
            int d = getMaxDepthHelper(l, depth + 1);
            if (d > maxDepth) {
                maxDepth = d;
            }
        }
        return maxDepth;
    }

    public int getMaxDepth(LoopNode node) {
        return getMaxDepthHelper(node, 0);
    }

    ArrayList<LERStatement> minimumUnion() {
        ArrayList<LERStatement> res = new ArrayList<LERStatement>();
        // List of trees
        ArrayList<LoopNode> worklist = generateWorklist();
        ArrayList<LoopNode> forest = new ArrayList<LoopNode>();
        while (!worklist.isEmpty()) {
            LoopNode minNode = getMinimumCost(worklist);
            forest.add(minNode);
            worklist.remove(minNode);
            if (!worklist.isEmpty()) {
                for (LoopNode l : worklist) {
                    if (minNode.relloopids != null) {
                        if (minNode.relloopids.contains(l.node.id)) {
                            minNode.addParent(l);
                            l.cost /= minNode.node.getCost();
                        }
                    }
                }
                minNode.confirmParent();
            }
        }
        //printForest(forest);
        if (forest.isEmpty()) {
            res.add(this);
            return res;
        }
        ArrayList<Operand> availOperands = getAvailOperands();
        ArrayList<LoopNode> trees = getForestRoots(forest);
        ArrayList<Operand> topResOperands = new ArrayList<Operand>();
        for (LoopNode tree : trees) {
            int maxDepth = getMaxDepth(tree);
            if (maxDepth == 0 && tree.relloopids.isEmpty()) {
                LERStatement newStatement = new LERStatement();
                newStatement.EncapsulatedLoops = new ArrayList<Loop>();
                newStatement.E = new ArrayList<Operand>();
                Operand boundOp = new Operand();
                boundOp.operandType = OperandType.CONSTANT;
                boundOp.value = tree.node.getCost();
                if (boundOp.value == 0) {
                    boundOp.operandType = OperandType.SCALAR;
                    boundOp.id = tree.node.uBound;
                }
                newStatement.E.add(boundOp);
                newStatement.EType = EType;
                String resName = "tmp" + Glory.tmpNames++;
                Operand resOperand = new Operand();
                resOperand.id = resName;
                resOperand.operandType = OperandType.SCALAR;
                newStatement.R = resOperand;
                res.add(newStatement);
                tree.result = resOperand;
            } else {
                for (int i = maxDepth; i >= 0; i--) {
                    ArrayList<LoopNode> depthList = getDepthList(tree, i);
                    for (LoopNode ln : depthList) {
                        LERStatement newStatement = new LERStatement();
                        newStatement.EncapsulatedLoops = new ArrayList<Loop>();
                        newStatement.E = new ArrayList<Operand>();
                        newStatement.EType = EType;
                        ArrayList<String> encapResIndices = new ArrayList<String>();
                        ArrayList<String> resIndices = new ArrayList<String>();
                        String dependence_id = null;

                        HashSet<String> eRelloopids = new HashSet<>();
                        // Build "E" of "L E = R"
                        // Add result operands from the children nodes
                        if (ln.children != null) {
                            for (LoopNode child : ln.children) {
                                if (child.result != null) {
                                    newStatement.E.add(child.result);
                                    if (child.result.encapsulated_abstract_index != null) {
                                        eRelloopids.addAll(child.result.encapsulated_abstract_index);
                                    }
                                } else {
                                    System.out.println("Error: Child result is null");
                                    assert (false);
                                }
                            }
                        }
                        // Add operands from remaining list
                        for (Operand o : availOperands) {
                            if (o.operandType == OperandType.ARRAY) {
                                if (o.encapsulated_abstract_index.contains(ln.node.id)) {
                                    newStatement.E.add(o);
                                    if (o.encapsulated_abstract_index != null) {
                                        eRelloopids.addAll(o.encapsulated_abstract_index);
                                    }
                                    if (o.dependence != null) {
                                        eRelloopids.add(o.dependence);
                                    }
                                } else if (o.dependence != null) {
                                    if (ln.relloopids.contains(o.dependence)) {
                                        newStatement.E.add(o);
                                        if (o.encapsulated_abstract_index != null) {
                                            eRelloopids.addAll(o.encapsulated_abstract_index);
                                        }
                                        eRelloopids.add(o.dependence);
                                    }
                                }
                            } else {
                                if (o.dependence != null) {
                                    if (ln.relloopids.contains(o.dependence)) {
                                        // System.out.println(o.toString());
                                        newStatement.E.add(o);
                                    }
                                }
                            }
                        }
                        availOperands.removeAll(newStatement.E);

                        // Build "L" of "L E = R"
                        for (Loop l : EncapsulatedLoops) {
                            if (l.loopType == LoopType.SUMMATION) {
                                if (l.id.equals(ln.node.id)) {
                                    newStatement.EncapsulatedLoops.add(l);
                                    if (l.regloopid != null) {
                                        resIndices.add(l.regloopid);
                                        encapResIndices.add(l.id);
                                    }
                                } else if (eRelloopids.contains(l.id)) {
                                    Loop newLoop = new Loop();
                                    newLoop.loopType = LoopType.REGULAR;
                                    newLoop.id = l.id;
                                    newLoop.lBound = l.lBound;
                                    newLoop.uBound = l.uBound;
                                    newLoop.regloopid = l.regloopid;
                                    newLoop.leftLoop = l.leftLoop;
                                    newLoop.rightLoop = l.rightLoop;
                                    newStatement.EncapsulatedLoops.add(newLoop);
                                    encapResIndices.add(l.id);
                                    if (l.regloopid != null) {
                                        resIndices.add(l.regloopid);
                                    } else {
                                        resIndices.add(l.id);
                                    }
                                    dependence_id = l.id;
                                }
                            } else if (l.loopType == LoopType.REGULAR) {
                                if (eRelloopids.contains(l.id)) {
                                    newStatement.EncapsulatedLoops.add(l);
                                    encapResIndices.add(l.id);
                                    if (l.regloopid != null) {
                                        resIndices.add(l.regloopid);
                                    } else {
                                        resIndices.add(l.id);
                                    }
                                    dependence_id = l.id;
                                }
                            }
                        }

                        // Build "R" of "L E = R"
                        String resName = "tmp" + Glory.tmpNames++;
                        Operand resOperand = new Operand();
                        resOperand.id = resName;
                        if (!resIndices.isEmpty()) {
                            resOperand.operandType = OperandType.ARRAY;
                            resOperand.abstract_index = new ArrayList<String>(resIndices);
                            resOperand.encapsulated_abstract_index = new ArrayList<String>(encapResIndices);
                        } else {
                            resOperand.operandType = OperandType.SCALAR;
                        }
                        resOperand.dependence = dependence_id;

                        ln.result = resOperand;
                        newStatement.R = resOperand;
                        res.add(newStatement);
                    }
                }
            }
        }
        LERStatement topStatement = new LERStatement();
        HashSet<String> topResIndices = new HashSet<String>();
        HashSet<String> topEncapIndices = new HashSet<String>();
        topStatement.EncapsulatedLoops = new ArrayList<Loop>();
        for (LoopNode l : trees) {
            topResOperands.add(l.result);
        }
        topStatement.E = new ArrayList<Operand>(topResOperands);
        topStatement.E.addAll(availOperands);
        for (Operand o : topStatement.E) {
            if (o.abstract_index != null) {
                topResIndices.addAll(o.abstract_index);
            }
            if (o.encapsulated_abstract_index != null) {
                topEncapIndices.addAll(o.encapsulated_abstract_index);
            }
            if (o.dependence != null) {
                topResIndices.add(o.dependence);
                topEncapIndices.add(o.dependence);
            }
        }
        for (Loop l : EncapsulatedLoops) {
            // System.out.println("Encap Loop Eval: " + l.toString());
            if (l.loopType != LoopType.SUMMATION) {
                topStatement.EncapsulatedLoops.add(l);
            } else if (l.loopType == LoopType.SUMMATION) {
                // System.out.println("Summation Loop Eval: " + l.toString());
                if (l.regloopid != null) {
                    if (topResIndices.contains(l.regloopid)) {
                        // System.out.println("Encap Summation Loop Eval: " + l.toString());
                        // Make new regular encapsulated loop.
                        Loop newLoop = new Loop();
                        newLoop.loopType = LoopType.REGULAR;
                        newLoop.id = l.id;
                        newLoop.lBound = l.lBound;
                        newLoop.uBound = l.uBound;
                        newLoop.oldloopid = l.oldloopid;
                        Loop leftLoop = new Loop();
                        leftLoop.loopType = LoopType.REGULAR;
                        leftLoop.id = l.leftLoop.id;
                        leftLoop.lBound = l.leftLoop.lBound;
                        leftLoop.uBound = l.leftLoop.uBound;
                        leftLoop.oldloopid = l.leftLoop.oldloopid;
                        newLoop.leftLoop = leftLoop;
                        Loop rightLoop = new Loop();
                        rightLoop.loopType = LoopType.REGULAR;
                        rightLoop.id = l.rightLoop.id;
                        rightLoop.lBound = l.rightLoop.lBound;
                        rightLoop.uBound = l.rightLoop.uBound;
                        rightLoop.oldloopid = l.rightLoop.oldloopid;
                        newLoop.rightLoop = rightLoop;
                        topStatement.EncapsulatedLoops.add(newLoop);
                    }
                } else if (topEncapIndices.contains(l.id)) {
                    Loop newLoop = new Loop();
                    newLoop.loopType = LoopType.REGULAR;
                    newLoop.id = l.id;
                    newLoop.lBound = l.lBound;
                    newLoop.uBound = l.uBound;
                    topStatement.EncapsulatedLoops.add(newLoop);
                }
            }
        }
        topStatement.EType = EType;
        topStatement.R = R;
        res.add(topStatement);
        return res;
    }

    public class operandNode {
        public Operand operand;
        public HashSet<String> relLoopIds;
        public ArrayList<operandNode> children;
        public operandNode parent;

        public operandNode(Operand operand) {
            this.operand = operand;
            children = new ArrayList<operandNode>();
            parent = null;
            if (operand.encapsulated_abstract_index != null) {
                relLoopIds = new HashSet<String>(operand.encapsulated_abstract_index);
            } else {
                relLoopIds = new HashSet<String>();
            }
            if (operand.dependence != null) {
                relLoopIds.add(operand.dependence);
            }
        }

        public void addParent(operandNode parent) {
            this.parent = parent;
            parent.children.add(this);
        }

        public int getRelLoopSize() {
            return relLoopIds.size();
        }

        public void addToTree(ArrayList<operandNode> tree) {
            assert (!tree.isEmpty());
            // Find the smallest sized parent node to add the node to
            // The relLoopIds of the child should be a subset of the parent
            int minSize = -1;
            operandNode minNode = null;
            for (operandNode n : tree) {
                if (this.relLoopIds != null) {
                    if (n.relLoopIds.containsAll(this.relLoopIds)) {
                        if (n.relLoopIds.size() <= minSize || minSize == -1) {
                            minSize = n.relLoopIds.size();
                            minNode = n;
                        }
                    }
                } else {
                    if (n.relLoopIds.size() <= minSize || minSize == -1) {
                        minSize = n.relLoopIds.size();
                        minNode = n;
                    }
                }
            }
            if (minNode != null) {
                this.addParent(minNode);
            } else {
                System.out.println("Error: Could not find a parent node for " + this.operand.toEncapsulatedString());
                assert (false);
            }
        }

        public int getIndexSpace(HashSet<String> seenIndexSet) {
            int indexSpace = 0;
            if (operand.operandType == OperandType.ARRAY) {
                for (String i : operand.encapsulated_abstract_index) {
                    if (!seenIndexSet.contains(i)) {
                        for (Loop l : EncapsulatedLoops) {
                            if (l.id.equals(i)) {
                                if (indexSpace == 0) {
                                    indexSpace = l.getCost();
                                } else {
                                    indexSpace *= l.getCost();
                                }
                            }
                        }
                    }
                }
            }
            return indexSpace;
        }
    }

    public operandNode getMaxRelLoopSize(ArrayList<operandNode> worklist) {
        int maxRelLoopSize = -1;
        operandNode maxNode = null;
        for (operandNode node : worklist) {
            if (node.getRelLoopSize() > maxRelLoopSize) {
                maxRelLoopSize = node.getRelLoopSize();
                maxNode = node;
            }
        }
        return maxNode;
    }

    public void printClosureTree(operandNode node, int depth) {
        for (int i = 0; i < depth; i++) {
            System.out.print("  ");
        }
        System.out.println(node.operand.toEncapsulatedString());
        for (operandNode n : node.children) {
            printClosureTree(n, depth + 1);
        }
    }

    public int getMaxDepth(operandNode node) {
        if (node.children.isEmpty()) {
            return 0;
        }
        int maxDepth = 0;
        for (operandNode n : node.children) {
            int d = getMaxDepth(n) + 1;
            if (d > maxDepth) {
                maxDepth = d;
            }
        }
        return maxDepth;
    }

    public int getMaxDepthHelper(operandNode node, int depth) {
        if (node.children.isEmpty()) {
            return depth;
        }
        int maxDepth = 0;
        for (operandNode n : node.children) {
            int d = getMaxDepthHelper(n, depth + 1);
            if (d > maxDepth) {
                maxDepth = d;
            }
        }
        return maxDepth;
    }

    public ArrayList<operandNode> getDepthListHelper(operandNode node, int depth, int depthCount) {
        ArrayList<operandNode> res = new ArrayList<operandNode>();
        if (depthCount == depth) {
            res.add(node);
        }
        if (node.children.isEmpty()) {
            return res;
        } else {
            for (operandNode l : node.children) {
                res.addAll(getDepthListHelper(l, depth, depthCount + 1));
            }
        }
        return res;
    }

    public ArrayList<operandNode> getDepthList(operandNode node, int depth) {
        ArrayList<operandNode> res = new ArrayList<operandNode>();
        int depthCount = 0;
        if (depthCount == depth) {
            res.add(node);
        }
        if (node.children.isEmpty()) {
            return res;
        } else {
            for (operandNode l : node.children) {
                res.addAll(getDepthListHelper(l, depth, depthCount + 1));
            }
        }
        return res;
    }

    public ArrayList<LERStatement> operandClosure() {
        ArrayList<LERStatement> res = new ArrayList<LERStatement>();
        if (EncapsulatedLoops.isEmpty()) {
            res.add(this);
            return res;
        }
        ArrayList<operandNode> worklist = new ArrayList<operandNode>();
        for (Operand e : E) {
            operandNode node = new operandNode(e);
            worklist.add(node);
        }
        Operand rootOperand = new Operand();
        rootOperand.encapsulated_abstract_index = new ArrayList<String>();
        for (Loop l : EncapsulatedLoops) {
            // if(l.loopType == LoopType.REGULAR){
            // rootOperand.encapsulated_abstract_index.add(l.id);
            // }
            rootOperand.encapsulated_abstract_index.add(l.id);
        }
        rootOperand.id = "RootOperand";
        rootOperand.operandType = OperandType.ARRAY;
        operandNode rootNode = new operandNode(rootOperand);
        ArrayList<operandNode> closureTree = new ArrayList<operandNode>();
        closureTree.add(rootNode);
        while (!worklist.isEmpty()) {
            operandNode maxNode = getMaxRelLoopSize(worklist);
            // System.out.println("Max node: " + maxNode.operand.toEncapsulatedString());
            worklist.remove(maxNode);
            maxNode.addToTree(closureTree);
            closureTree.add(maxNode);
        }
        // Print the closure tree, starting with rootOperand
        //System.out.println("------  Closure Tree  ------");
        //printClosureTree(rootNode, 0);

        // Run the closure-based algorithm to decide post-order traversal.

        int closureDepth = getMaxDepth(rootNode);
        for (int i = closureDepth; i >= 0; i--) {
            if (i == 0 && res.isEmpty()) {
                res.add(this);
                return res;
            } else if (i == 0) {
                res.get(res.size() - 1).R = this.R;
                Glory.tmpNames--;
                return res;
            }
            ArrayList<operandNode> depthList = getDepthList(rootNode, i);
            ArrayList<operandNode> visitedList = new ArrayList<operandNode>();
            for (operandNode n : depthList) {
                if (visitedList != null) {
                    if (visitedList.contains(n)) {
                        continue;
                    }
                }
                ArrayList<operandNode> siblings = new ArrayList<operandNode>();
                if (n.parent != null) {
                    siblings = n.parent.children;
                } else {
                    siblings.add(n);
                }
                HashSet<String> seenSet = new HashSet<String>();
                while (!siblings.isEmpty()) {
                    // Get smallest indexSpace
                    int minIndexSpace = Integer.MAX_VALUE;
                    operandNode minNode = null;
                    for (operandNode s : siblings) {
                        int indexSpace = s.getIndexSpace(seenSet);
                        if (indexSpace < minIndexSpace) {
                            minIndexSpace = indexSpace;
                            minNode = s;
                        }
                    }
                    assert (minNode != null);
                    visitedList.add(minNode);
                    siblings.remove(minNode);
                    // Add the node to the result list

                    if (minNode.parent.operand.id.equals("RootOperand")) {
                        // Replace the root operand with minNode
                        minNode.parent.operand = minNode.operand;
                    } else {
                        LERStatement newStatement = new LERStatement();
                        newStatement.EncapsulatedLoops = new ArrayList<Loop>();
                        newStatement.E = new ArrayList<Operand>();
                        newStatement.E.add(minNode.operand);
                        newStatement.E.add(minNode.parent.operand);
                        newStatement.EType = EType;
                        ArrayList<String> eRelLoops = new ArrayList<String>();
                        if (minNode.operand.encapsulated_abstract_index != null) {
                            eRelLoops.addAll(minNode.operand.encapsulated_abstract_index);
                        }
                        if (minNode.parent.operand.encapsulated_abstract_index != null) {
                            eRelLoops.addAll(minNode.parent.operand.encapsulated_abstract_index);
                        }
                        if (minNode.operand.dependence != null) {
                            eRelLoops.add(minNode.operand.dependence);
                        }
                        if (minNode.parent.operand.dependence != null) {
                            eRelLoops.add(minNode.parent.operand.dependence);
                        }
                        ArrayList<String> resIndices = new ArrayList<String>();
                        ArrayList<String> encapResIndices = new ArrayList<String>();
                        String dependence_id = null;
                        // Create the "L"
                        for (Loop l : EncapsulatedLoops) {
                            if (l.loopType == LoopType.SUMMATION) {
                                if (eRelLoops.contains(l.id)) {
                                    newStatement.EncapsulatedLoops.add(l);
                                    encapResIndices.add(l.id);
                                    if (l.regloopid != null) {
                                        resIndices.add(l.regloopid);
                                    } else {
                                        resIndices.add(l.id);
                                    }
                                    dependence_id = l.id;
                                }
                            } else if (l.loopType == LoopType.REGULAR) {
                                if (eRelLoops.contains(l.id)) {
                                    newStatement.EncapsulatedLoops.add(l);
                                    encapResIndices.add(l.id);
                                    if (l.regloopid != null) {
                                        resIndices.add(l.regloopid);
                                    } else {
                                        resIndices.add(l.id);
                                    }
                                    dependence_id = l.id;
                                }
                            }
                        }
                        // Create the "R"
                        String resName = "tmp" + Glory.tmpNames++;
                        Operand resOperand = new Operand();
                        resOperand.id = resName;
                        if (!resIndices.isEmpty()) {
                            resOperand.operandType = OperandType.ARRAY;
                            resOperand.abstract_index = new ArrayList<String>(resIndices);
                            resOperand.encapsulated_abstract_index = new ArrayList<String>(encapResIndices);
                        } else {
                            resOperand.operandType = OperandType.SCALAR;
                        }
                        resOperand.dependence = dependence_id;
                        newStatement.R = resOperand;
                        minNode.parent.operand = resOperand;
                        res.add(newStatement);
                    }
                    if (minNode.operand.encapsulated_abstract_index != null) {
                        seenSet.addAll(minNode.operand.encapsulated_abstract_index);
                    }
                    if (minNode.operand.dependence != null) {
                        seenSet.add(minNode.operand.dependence);
                    }
                }
                // LERStatement newStatement = new LERStatement();
                // newStatement.EncapsulatedLoops = new ArrayList<Loop>();
                // newStatement.E = new ArrayList<Operand>();
                // newStatement.EType = EType;
                // ArrayList<String> encapResIndices = new ArrayList<String>();
                // ArrayList<String> resIndices = new ArrayList<String>();
                // String dependence_id = null;

                // HashSet <String> eRelloopids = new HashSet<>();
                // // Build "E" of "L E = R"
                // // Add result operands from the children nodes
                // if (n.children != null){
                // for(operandNode child : n.children){
                // if(child.operand != null){
                // newStatement.E.add(child.operand);
                // if(child.operand.encapsulated_abstract_index != null){
                // eRelloopids.addAll(child.operand.encapsulated_abstract_index);
                // }
                // } else{
                // System.out.println("Error: Child result is null");
                // assert(false);
                // }
                // }
                // }
                // // Add operands from remaining list
                // for(Operand o : E){
                // if(o.operandType == OperandType.ARRAY){
                // if(o.encapsulated_abstract_index.containsAll(n.relLoopIds)){
                // newStatement.E.add(o);
                // if(o.encapsulated_abstract_index != null){
                // eRelloopids.addAll(o.encapsulated_abstract_index);
                // }
                // if(o.dependence != null){
                // eRelloopids.add(o.dependence);
                // }
                // } else if (o.dependence != null){
                // if(n.relLoopIds.contains(o.dependence)){
                // newStatement.E.add(o);
                // if(o.encapsulated_abstract_index != null){
                // eRelloopids.addAll(o.encapsulated_abstract_index);
                // }
                // eRelloopids.add(o.dependence);
                // }
                // }
                // } else{
                // if(o.dependence != null){
                // if(n.relLoopIds.contains(o.dependence)){
                // newStatement.E.add(o);
                // }
                // }
                // }
                // }
                // E.removeAll(newStatement.E);

                // // Build "L" of "L E = R"
                // for(Loop l : EncapsulatedLoops){
                // if(l.loopType == LoopType.SUMMATION){
            }
        }
        return res;
    }

    public void decapsulateLoops() {
        try {
            // Encapsulate the loops
            L.clear();
            for (Loop enl : EncapsulatedLoops) {
                if (enl.leftLoop == null) {
                    L.add(enl);
                } else {
                    L.add(enl.leftLoop);
                    L.add(enl.rightLoop);
                }
            }
        } catch (Exception e) {
            System.out.println("Error in loopDecapsulation: " + e);
            e.printStackTrace();
        }
    }

    public LERStatement partiallyLoopInvariant() {
        try {
            // If last EncapsulatedLoops has a non-null right loop, then continue.
            // Otherwise, return this.
            if (EncapsulatedLoops.size() == 0) {
                return this;
            }
            if (EncapsulatedLoops.get(EncapsulatedLoops.size() - 1).rightLoop != null) {
                if (EncapsulatedLoops.get(EncapsulatedLoops.size() - 1).rightLoop.loopType == LoopType.REGULAR) {
                    return this;
                }
                // Create a new LERStatement
                LERStatement newStatement = new LERStatement();
                newStatement.L = new ArrayList<Loop>();
                // Add all but the last loop
                for (int i = 0; i < EncapsulatedLoops.size(); i++) {
                    newStatement.L.add(L.get(i));
                }
                String flagged_id = L.get(L.size() - 1).id;
                newStatement.EncapsulatedLoops = new ArrayList<Loop>();
                newStatement.E = new ArrayList<Operand>();
                boolean incremental_flag = false;
                for (Operand o : E) {
                    if (o.operandType == OperandType.ARRAY) {
                        if (o.abstract_index.contains(flagged_id)) {
                            o.operandType = OperandType.INCREMENTAL;
                            o.flagged_id = flagged_id;
                            o.new_id = L.get(L.size() - 1).uBound;
                            newStatement.E.add(o);
                            incremental_flag = true;
                        } else {
                            newStatement.E.add(o);
                        }
                    } else {
                        newStatement.E.add(o);
                    }
                }
                if (incremental_flag == false) {
                    Operand newConstOp = new Operand();
                    newConstOp.operandType = OperandType.INCREMENTAL;
                    newConstOp.flagged_id = flagged_id;
                    newConstOp.new_id = L.get(L.size() - 1).uBound;
                    newConstOp.value = 1;
                    newStatement.E.add(newConstOp);
                }
                newStatement.EType = EType;
                newStatement.R = R;
                return newStatement;
            } else {
                return this;
            }
        } catch (Exception e) {
            System.out.println("Error in loopInvariantLoops: " + e);
            e.printStackTrace();
        }
        return null;
    }

    ArrayList<Loop> L = new ArrayList<Loop>();
    ArrayList<Loop> EncapsulatedLoops = new ArrayList<Loop>();
    ArrayList<Operand> E = new ArrayList<Operand>();
    Operand R;
    ExpressionType EType;
}

// func/*/stat, -> all stat nodes grandkids of any func node
// e/expression/term/factor
// expr/primary/!ID, -> anything but ID under primary under any expr node

// $ antlr4-parse Expr.g4 prog -gui
// 10+20*30
// ^D
