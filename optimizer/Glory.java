package optimizer;

import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CommonTokenStream;
// import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.antlr.v4.runtime.tree.xpath.XPath;
import org.antlr.v4.runtime.tree.*;

import java.io.FileInputStream;
import java.io.InputStream;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Collection;

public class Glory {
    static int tmpNames = 0;
    public static void main(String[] args) throws Exception {

        ANTLRInputStream inputStream = new ANTLRInputStream(
                new FileInputStream(args[0]));

        try {
            // Get our lexer
            GloryLexer lexer = new GloryLexer(inputStream);
            // Get a list of matched tokens
            CommonTokenStream tokens = new CommonTokenStream(lexer);

            // Pass the tokens to the parsercatcat
            GloryParser parser = new GloryParser(tokens);

            // Specify our entry point
            GloryParser.StatementContext drinkSentenceContext = parser.statement();


            // Walk it and attach our listener
            ParseTreeWalker walker = new ParseTreeWalker();
            // DirectiveListener listener = new DirectiveListener();
            // walker.walk(listener, drinkSentenceContext);
            LERListener lerlistener = new LERListener();
            walker.walk(lerlistener, drinkSentenceContext);
            LERStatement lerstatement = lerlistener.getLERStatement();
            String xpath_factor = "//term/factor";
            ParseTree r = (ParseTree) drinkSentenceContext.r();
            int text_offset = 1;
            if (r.getChildCount() == 5) {
                text_offset = 2;
            }
            if (r.getChildCount() == 4 || r.getChildCount() == 5) {
                // Check for array access
                if (r.getChild(text_offset).getText().equals("[")) {
                    // This is an array access
                    ParseTree indexExpression = r.getChild(text_offset+1);
                    // Find the factors within this expression
                    ArrayList<String> relevantIndices = new ArrayList<String>();
                    for (ParseTree fa : XPath.findAll(indexExpression, xpath_factor, parser)) {
                        GloryParser.FactorContext factor = (GloryParser.FactorContext) fa;
                        if(factor.id() != null){
                            relevantIndices.add(factor.id().getText());
                        }
                    }
                    System.out.println("Relevant indices: " + relevantIndices.toString());
                    lerstatement.R.abstract_index = relevantIndices;
                }
            }

            // Formula Simplification
            ArrayList<LERStatement.Operand> tmpRes = new ArrayList<LERStatement.Operand>();
            ArrayList<LERStatement> lerstatementlist = new ArrayList<LERStatement>();
            // Find all ///expressionPrime/expression
            String xpath = "//expressionPrime/expression/term";
            Collection<ParseTree> terms = new ArrayList<ParseTree>();
            ParseTree currExpr = (ParseTree) drinkSentenceContext.e().expression();
            while(currExpr.getChildCount() == 2){
                terms.add((ParseTree) currExpr.getChild(0));
                currExpr = currExpr.getChild(1);
                if(currExpr.getChildCount() == 2){
                    currExpr = currExpr.getChild(1);
                }
            }
            // terms.add((ParseTree) drinkSentenceContext.e().expression().term());
            // terms.addAll(XPath.findAll(drinkSentenceContext.e(), xpath, parser));
            for (ParseTree t : terms){
                ArrayList<GloryParser.FactorContext> topFactorList = getTopFactors((GloryParser.TermContext) t);
                ArrayList<LERStatement.Operand> factorOperands = getFactorOperands(topFactorList, parser, lerstatement.L);                
                // Recursively traverse t to get the top level factors
                LERStatement newlerstatement = new LERStatement();
                newlerstatement.L = new ArrayList<>();
                for (LERStatement.Loop loop : lerstatement.L) {
                    LERStatement.Loop newLoop = new LERStatement.Loop();
                    newLoop.id = loop.id;
                    newLoop.lBound = loop.lBound;
                    newLoop.uBound = loop.uBound;
                    newLoop.loopType = loop.loopType;
                    newLoop.regloopid = loop.regloopid;
                    newlerstatement.L.add(newLoop);
                }
                newlerstatement.E = factorOperands;
                newlerstatement.EType = LERStatement.ExpressionType.MUL;
                HashSet<String> relevantLoops = new HashSet<String>();
                if(lerstatement.R.index != null && lerstatement.R.abstract_index != null){
                    for (String loopid : lerstatement.R.abstract_index){
                        relevantLoops.add(loopid);
                    }
                }
                int idcounter = 0;
                for (ParseTree t_f : XPath.findAll(t, xpath_factor, parser) ){
                    // System.out.println(t.getText());
                    // Check for number of children
                    int arr_offset = 1;
                    if (t_f.getChildCount() == 5) {
                        arr_offset = 2;
                    }
                    if (t_f.getChildCount() == 4 || t_f.getChildCount() == 5) {
                        // Check for array access
                        if (t_f.getChild(arr_offset).getText().equals("[")) {
                            // This is an array access
                            String arrayName = t_f.getChild(0).getText();
                            GloryParser.IdContext id = (GloryParser.IdContext) t_f.getChild(0);
                            ParseTree indexExpression = t_f.getChild(arr_offset+1);
                            // Find the factors within this expression
                            for (ParseTree fa : XPath.findAll(indexExpression, xpath_factor, parser)) {
                                GloryParser.FactorContext factor = (GloryParser.FactorContext) fa;
                                if(factor.id() != null){
                                    relevantLoops.add(factor.getText());
                                    idcounter++;
                                }
                            }
                        }
                    }
                }
                // Grab all subscript ids
                for (ParseTree t_id : XPath.findAll(t, "//subscript/id", parser) ){
                    GloryParser.IdContext id = (GloryParser.IdContext) t_id;
                    if (id != null){
                        relevantLoops.add(id.getText());
                    }
                }
                ArrayList<String> outputIndices = new ArrayList<String>();
                // Traverse loops in the LERStatement, and check if they are relevant
                for (LERStatement.Loop loop : lerstatement.L){
                    if (relevantLoops.contains(loop.id) && loop.loopType != LERStatement.LoopType.SUMMATION){
                        outputIndices.add(loop.id);
                    }
                    else if (relevantLoops.contains(loop.id) && loop.loopType == LERStatement.LoopType.SUMMATION){
                        idcounter--;
                    }
                }
                LERStatement tmpLER = new LERStatement();
                LERStatement.Operand tmpOperand = tmpLER.new Operand();
                if(idcounter == 0){
                    tmpOperand.operandType = LERStatement.OperandType.SCALAR;
                    tmpOperand.value = 0;
                    tmpOperand.id = "tmp" + tmpNames;
                    if(outputIndices.size() > 0){
                        tmpOperand.dependence = outputIndices.get(0);
                    }
                    tmpNames++;
                } else {
                    tmpOperand.operandType = LERStatement.OperandType.ARRAY;
                    tmpOperand.id = "tmp" + tmpNames;
                    tmpNames++;
                    if(outputIndices.size() > 0){
                        tmpOperand.dependence = outputIndices.get(0);
                    }
                    tmpOperand.index = null;
                    tmpOperand.abstract_index = outputIndices;
                }
                tmpRes.add(tmpOperand);
                newlerstatement.R = tmpOperand;
                lerstatementlist.add(newlerstatement);
            }
            if(lerstatementlist.size() > 1){
                lerstatement.E = tmpRes;
                lerstatement.EType = LERStatement.ExpressionType.ADD;
                // Convert all summation in L to regular
                for (LERStatement.Loop loop : lerstatement.L){
                    if (loop.loopType == LERStatement.LoopType.SUMMATION){
                        loop.loopType = LERStatement.LoopType.REGULAR;
                    }
                }
                lerstatementlist.add(lerstatement);
            }
            else{
                lerstatementlist.get(0).R = lerstatement.R;
            }
            //System.out.println();
            //// print ler statements
            //System.out.println("----------------------------------");
            //System.out.println("      Formula Simplification      ");
            //System.out.println("----------------------------------");
            //for (LERStatement l : lerstatementlist){
            //    System.out.println(l.toString());
            //}
            //System.out.println();
            //System.out.println("-------------------------------");
            //System.out.println("      Operand Abstraction      ");
            //System.out.println("-------------------------------");
            //for (LERStatement l : lerstatementlist){
            //    System.out.println(l.toAbstractString());
            //}
            //
            //System.out.println();
            //System.out.println("------------------------------");
            //System.out.println("      Loop Encapsulation      ");
            //System.out.println("------------------------------");
            //
            // Loop encapsulation
            for (LERStatement l : lerstatementlist){
                l.encapsulateLoops();
                //System.out.println(l.toEncapsulatedString());
            }
            
            //System.out.println();
            //System.out.println("-----------------------------------");
            //System.out.println("      Minimum Union Algorithm      ");
            //System.out.println("-----------------------------------");
			
            // Minimum Union Algorithm
            ArrayList<LERStatement> minimumunionlerstatement = new ArrayList<LERStatement>();
            for (LERStatement l : lerstatementlist){
                // l.minimumUnion(tmpNames);
                minimumunionlerstatement.addAll(l.minimumUnion());
            }

            //System.out.println();
            //System.out.println("--------------------------------------------");
            //System.out.println("      Remove Redundant Reduction Loops      ");
            //System.out.println("--------------------------------------------");
            //
            //for (LERStatement l : minimumunionlerstatement){
            //    System.out.println(l.toEncapsulatedString());
            //}
            //
            //System.out.println();
            //System.out.println("-------------------------------------");
            //System.out.println("      Operand Closure Algorithm      ");
            //System.out.println("-------------------------------------");

            // Redundant Regular Loops
            ArrayList<LERStatement> redundantregularlerstatement = new ArrayList<LERStatement>();
            for (LERStatement l : minimumunionlerstatement){
                redundantregularlerstatement.addAll(l.operandClosure());
                // l.operandClosure(tmpNames);
            }

            //System.out.println();
            //System.out.println("------------------------------------------");
            //System.out.println("      Remove Redundant Regular Loops      ");
            //System.out.println("------------------------------------------");
            //
            //for (LERStatement l : redundantregularlerstatement){
            //    System.out.println(l.toEncapsulatedString());
            //}
            //
            //System.out.println();
            //System.out.println("------------------------------");
            //System.out.println("      Loop Decapsulation      ");
            //System.out.println("------------------------------");
			
            for (LERStatement l : redundantregularlerstatement){
                l.decapsulateLoops();
                //System.out.println(l.toAbstractString());
            }

            //System.out.println();
            //System.out.println("-----------------------------------------");
            //System.out.println("      Partially Loop-Invariant Loops     ");
            //System.out.println("-----------------------------------------");
            
            ArrayList<LERStatement> partiallyloopinvariantlist = new ArrayList<LERStatement>();
            for (LERStatement l : redundantregularlerstatement){
                partiallyloopinvariantlist.add(l.partiallyLoopInvariant());
            }
            
            //for (LERStatement l : partiallyloopinvariantlist){
            //    System.out.println(l.toAbstractString());
            //}
            //
            //System.out.println();
            //System.out.println("----------------------------------");
            //System.out.println("      Operator Concretization     ");
            //System.out.println("----------------------------------");


            for (LERStatement l : partiallyloopinvariantlist){
                System.out.println(l.toString());
            }



            // ArrayList<LERStatement> lerstatementlist = lerstatement.optimize();
        } catch (Exception e) {
            // Print exception
            e.printStackTrace();
            System.out.println("Invalid Input");
        }
    }

    public static ArrayList<GloryParser.FactorContext> getTopFactors(GloryParser.TermContext ctx){
        ArrayList<GloryParser.FactorContext> topFactorList = new ArrayList<GloryParser.FactorContext>();
        GloryParser.TermContext curr_term = ctx;
        while (curr_term.getChildCount() == 3){
            topFactorList.add((GloryParser.FactorContext) curr_term.getChild(2));
            curr_term = (GloryParser.TermContext) curr_term.getChild(0);
        }
        topFactorList.add((GloryParser.FactorContext) curr_term.getChild(0));
        return topFactorList;
    }

    public static ArrayList<LERStatement.Operand> getFactorOperands(ArrayList<GloryParser.FactorContext> topFactorList, GloryParser parser, ArrayList<LERStatement.Loop> L){
        ArrayList<LERStatement.Operand> tmpRes = new ArrayList<LERStatement.Operand>();
        for (GloryParser.FactorContext factor : topFactorList){
            LERStatement tmpLER = new LERStatement();
            LERStatement.Operand tmpOperand = tmpLER.new Operand();
            tmpOperand.id = factor.id().getText();
            if (factor.subscript() != null) {
                tmpOperand.dependence = factor.subscript().id().getText();
            }
            if (factor.exprList() != null || factor.expression() != null) {
                if(factor.exprList() != null){
                    tmpOperand.index = (ParseTree) factor.exprList();
                } else {
                    tmpOperand.index = (ParseTree) factor.expression();
                }
                tmpOperand.operandType = LERStatement.OperandType.ARRAY;
                ArrayList<String> relevantIndices = new ArrayList<String>();
                for (ParseTree t : XPath.findAll(tmpOperand.index, "//factor", parser)){
                    GloryParser.FactorContext factor_cast = (GloryParser.FactorContext) t;
                    if(factor_cast.id() != null){
                        relevantIndices.add(factor_cast.id().getText());
                    }
                }
                // Intersection of relevantIndices and L.id
                ArrayList<String> outputIndices = new ArrayList<String>();
                for (String loopid : relevantIndices){
                    for (LERStatement.Loop loop : L){
                        if (loop.id.equals(loopid)){
                            outputIndices.add(loopid);
                        }
                    }
                }
                tmpOperand.abstract_index = outputIndices;
            }
            else {
                tmpOperand.operandType = LERStatement.OperandType.SCALAR;
            }
            tmpRes.add(tmpOperand);
        }
        return tmpRes;
    }


}
