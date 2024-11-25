package optimizer;

import java.util.ArrayList;
import org.antlr.v4.runtime.tree.*;
import org.antlr.v4.runtime.tree.xpath.XPath;

public class LERListener extends GloryBaseListener {
    // STEPS
    // 1. Oeprand FoldeAlternating forms
    // 2. Operands Abractions
    // 3. Loop Encapsulation
    // 4. Loop Invariant redundant loop
    // - Fomular Simplification
    // - Minimum Union Algorithm
    // 5. Loop Invairant regular loop
    // - Closure-based algorithm
    // 6. Partially loop-invariant loops
    // - operand concretization

    LERStatement lerStatement = new LERStatement();

    public LERStatement getLERStatement() {
        return lerStatement;
    }

    public LERListener() {
    }

    @Override
    public void enterStatement(GloryParser.StatementContext ctx) {
        // System.out.println("Statement starts:\n");
    }

    @Override
    public void enterL(GloryParser.LContext ctx) {
        super.enterL(ctx);

        if (ctx.loopType == null) {
            return;
        }

        LERStatement.Loop loop = new LERStatement.Loop();
        if (ctx.loopType.getType() == GloryParser.SUMMATION) {
            loop.loopType = LERStatement.LoopType.SUMMATION;
            loop.id = ctx.forParam().id().getText();
            loop.lBound = ctx.forParam().lBound().getText();
            loop.uBound = ctx.forParam().uBound().getText();
        } else if (ctx.loopType.getType() == GloryParser.OTHER) {
            loop.loopType = LERStatement.LoopType.WHILE;
            loop.id = ctx.subscript().id().getText();
        } else if (ctx.loopType.getType() == GloryParser.NORMAL) {
            loop.loopType = LERStatement.LoopType.REGULAR;
            loop.id = ctx.forParam().id().getText();
            loop.lBound = ctx.forParam().lBound().getText();
            loop.uBound = ctx.forParam().uBound().getText();
        }
        lerStatement.L.add(loop);
        System.out.println("Loop " + loop.id + " is of type " + loop.loopType);
    }

    @Override
    public void enterE(GloryParser.EContext ctx) {
        super.enterE(ctx);

        // lerStatement.E = ctx.expression();
        
    }

    // @Override
    // public void enterFactor(GloryParser.FactorContext ctx) {
    //     // System.out.println("Factor: " + ctx.getText());
    //     if (ctx.getChildCount() == 4){
    //         // System.out.println("Candidate array: " + ctx.getText());
    //         if(ctx.getChild(1).getText().equals("[")) {
    //             System.out.println("Indexed array: " + ctx.getText());
    //             // This is an array access
    //             String arrayName = ctx.getChild(0).getText();
    //             ParseTree indexExpression = ctx.getChild(2);
    //             System.out.println("Array " + arrayName + " is indexed by " + indexExpression.getText());
    //         }
    //     }
    // }

    @Override
    public void enterR(GloryParser.RContext ctx) {
        super.enterR(ctx);

        LERStatement.Operand operand = lerStatement.new Operand();
        // System.out.println("R: " + ctx.getText());
        operand.id = ctx.id().getText();
        if (ctx.subscript() != null) {
            operand.dependence = ctx.subscript().id().getText();
        }
        if (ctx.exprList() != null) {
            operand.index = (ParseTree) ctx.exprList();
            operand.operandType = LERStatement.OperandType.ARRAY;
        }
        else {
            operand.operandType = LERStatement.OperandType.SCALAR;
        }
        String indexing = operand.index != null ? operand.index.getText() : null;
        System.out.println("R id: " + operand.id + " dependence: " + operand.dependence + " index: " + indexing);
        lerStatement.R = operand;
    }

    @Override
    public void exitStatement(GloryParser.StatementContext ctx) {
        super.exitStatement(ctx);

        // GloryParser.EContext e = ctx.e();
        // LERStatement.Expression expression = LERStatement.Expression(e.exprList());

        // GloryParser.RContext r = ctx.r();
        // LERStatement.Operand operand = new LERStatement.Operand();
        // operand.dependence = r.subscript.getText();
        // operand.id = r.id().getText();
        // operand.operandType = LERStatement.OperandType.ARRAY;
        // operand.index = LERStatement.Expression(r.exprList());
        // lerStatement.R = operand;
    }
}